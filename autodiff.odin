package autodiff

import "core:fmt"
import "core:math"
import "core:mem"
import "core:runtime"
import "core:slice"
import "core:strings"
import "core:testing"

// Use "Forward accumulation" because (simple implementations) are significantly more robust to arbitrary & complex calculations.
// This provides d_/dInput information, which is not as friendly as a dOutput/d_ "reverse accumulation" algorithm.
// As a result this will not be as efficient for large "tweak all inputs affecting Y output" (e.g. neural networks)
// Each variable may be a scalar or vector, with element-wise operations.

// TODO Operation procedures for add & multiply that can take list inputs

Var :: struct {
	graph: ^Graph, // store a pointer to the graph owning this variable so that it doesn't have to be specifed as a procedure argument
	id:    u64, // index in graph.vars[]
	size:  u64, // length of val and dval; since their length must be the same, save 8 bytes by storing size separately
	val:   [^]f64,
	dval:  [^]f64,
}

Op :: struct {
	inputs: [2]u64, // index in graph.vars[]
	output: u64, // index in graph.vars[]
	kind:   OpKind, // add, multiply, max, etc.
}

Graph :: struct {
	// Forward-accumulation dependencies built-in by the order of operations added to the graph.
	ops:       [dynamic]Op,
	vars:      [dynamic]Var,
	vars_name: [dynamic]string, // Separate from Var to reduce memory of computing uncertainty()

	// Storage for the graph, including raw variable data
	allocator: runtime.Allocator,
	arena:     mem.Arena,
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Usage

main :: proc() {

	// Cantilever beam of constant hollow circular cross section, with a point load applied somewhere along its length
	// Using common small-angle-approximation beam model
	graph := graph_init()

	x := var_linspace(graph, low = 1.5, high = 2.5, size = 10, name = "x") // (m) location of load along the beam, 10 options 
	L := var_copy(graph, []f64{3}, name = "L") // (m) total length of the the beam

	F := var_copy(graph, []f64{1000}, name = "F") // (N) applied load
	E := var_copy(graph, []f64{71e9}, name = "E") // (Pa) material stiffness
	rho := var_copy(graph, []f64{2810}, name = "rho") // (kg/m^3) material density

	r_out := var_copy(graph, []f64{0.030}, name = "r_out") // (m) outside radius of the tube
	r_in := var_copy(graph, []f64{0.015}, name = "r_in") // (m) inside radius of the tube

	I := mult(math.PI / 4.0, sub(power(r_out, 4.0), power(r_in, 4.0))) // (m^4) bending moment of inertia of the beam
	area := mult(math.PI, sub(power(r_out, 2.0), power(r_in, 2.0)), "area") // (m^2) cross sectional area of the beam

	mass := mult(rho, mult(L, area), "mass") // (kg) mass of the beam

	stress_max := mult(div(r_out, I), mult(F, x), "stress_max") // (Pa) maximum stress in the beam

	delta1 := div(mult(F, power(x, 3.0)), mult(3.0, mult(E, I)))
	delta2 := add(1, div(mult(3, sub(L, x)), mult(2, x)))
	delta := mult(delta1, delta2, "deflection_tip") // (m) deflection of the tip of the beam

	uncertainty_collect(graph, targets = []Var{stress_max, mass, delta}, knobs = []Var{r_out, r_in})

	graph_print(graph)
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Core Functions


// STEP 1: Init a compute graph with an allocator to hold memory for the data.
graph_init :: proc(memory_mb := 50, capacity_initial := 50, allocator := context.allocator) -> ^Graph {
	context.allocator = allocator
	graph: ^Graph = new(Graph)

	// create an allocator and allocate stuff for the graph
	data, backing_alloc_err := make([]u8, memory_mb * mem.Megabyte)
	if backing_alloc_err != nil {
		fmt.printf("[ERROR] Cannot allocate backing memory for graph arena: '%v'\n", backing_alloc_err)
	}
	mem.arena_init(&graph.arena, data)
	graph.allocator = mem.arena_allocator(&graph.arena)

	// track: mem.Tracking_Allocator
	// mem.tracking_allocator_init(&track, context.allocator)
	// graph.allocator = mem.tracking_allocator(&track)

	// put stuff on there to start with; try to bunch this up front so it is faster to jump over
	graph.vars = make([dynamic]Var, len = 0, cap = capacity_initial, allocator = graph.allocator)
	graph.ops = make([dynamic]Op, len = 0, cap = capacity_initial, allocator = graph.allocator)
	graph.vars_name = make([dynamic]string, len = 0, cap = capacity_initial, allocator = graph.allocator)

	return graph
}

graph_print :: proc(graph: ^Graph) {
	for var in graph.vars {
		fmt.printf("Var: %v\n", graph.vars_name[var.id])
		fmt.printf(" val : %v\n", var.val)
		fmt.printf(" dval: %v\n", var.dval)
	}
}


// STEP 2: Define (input) variables
// Always allocates memory for graph to own the data. 
// If size is provided, allocates an empty array. If val[] is provided, copies in the values.
var :: proc {
	var_reserve,
	var_copy,
}

var_reserve :: proc(graph: ^Graph, size: u64, name: string) -> Var {
	x := Var {
		graph = graph,
		id    = u64(len(graph.vars)),
		size = size,
	}

	// No data was provided to copy, allocate according to size
	assert(size != 0)
	x.val = make([^]f64, size, allocator = graph.allocator)
	x.dval = make([^]f64, x.size, allocator = graph.allocator)

	append(&graph.vars, x)
	append(&graph.vars_name, name)
	return x
}

var_copy :: proc(graph: ^Graph, val: []f64, name: string) -> Var {
	x := Var {
		graph = graph,
		id    = u64(len(graph.vars)),
		size = u64(len(val)),
	}

	// Data was provided; copy it in
	assert(len(val) != 0)
	x.val = make([^]f64, x.size, allocator = graph.allocator)
	copy(x.val[0:x.size], val)
	// for i in 0..<x.size{
		// x.val[i] = val[i]
	// }
	x.dval = make([^]f64, x.size, allocator = graph.allocator)

	append(&graph.vars, x)
	append(&graph.vars_name, name)
	return x
}

var_linspace :: proc(graph: ^Graph, low, high: f64, size: u64, name: string) -> Var {
	var := var_reserve(graph, size = size, name = name)
	assert(low < high)

	for i in 0 ..< size {
		var.val[i] = low + f64(i) * (high - low) / f64(size)
	}

	return var
}

// STEP 3: Add operations to the graph - see section of operation functions. 


// STEP 4: Get some answers
// Compute how all the variables affect the given variable
uncertainty_collect :: proc(graph: ^Graph, targets: []Var, knobs: []Var) {
	for knob in knobs {
		uncertainty(graph, knob)

		// now record effect on targets
		for target in targets {
			fmt.printf(
				"d'%v'/d'%v' = %v\n",
				graph.vars_name[target.id],
				graph.vars_name[knob.id],
				graph.vars[target.id].dval,
			)
		}
		fmt.printf("\n")
	}
}

// Report how variations in 'vary' change the network; recompute everything and 
// then allow the users to see the effect of that variable
// Assume this initial injected uncertainty is a scalar (not a slice)
uncertainty :: proc(graph: ^Graph, vary: Var, error: f64 = 1.0) {
	for var in &graph.vars {
		slice.fill(var.dval[0:var.size], 0)
	}
	slice.fill(graph.vars[vary.id].dval[0:vary.size], error)

	// Recompute the graph
	for op_id in 0 ..< len(graph.ops) {
		op := graph.ops[op_id]

		// each forward computation needs to compute uncertainty, of the output variable
		// re-do the calculation work
		switch op.kind {
		case .UNKNOWN:
			panic("Unset operation type!")

		case .ADD:
			add_(graph, op)
		case .SUBTRACT:
			sub_(graph, op)
		case .MULTIPLY:
			mult_(graph, op)
		case .DIVIDE:
			div_(graph, op)
		case .SINE:
			sin_(graph, op)
		case .COSINE:
			cos_(graph, op)
		case .TANGENT:
			tan_(graph, op)
		case .ATAN2:
			atan2_(graph, op)
		case .TANH:
			tanh_(graph, op)
		case .POWER:
			power_(graph, op)
		case .MIN:
			vmin_(graph, op)
		case .MAX:
			vmax_(graph, op)
		}
	}
}

// TODO some sensible way of copying out the dvals
// TODO procedure to update values of variables based on dvals

///////////////////////////////////////////////////////////////////////////////////////////////////
// The Operations

OpKind :: enum {
	UNKNOWN = 0,
	ADD,
	SUBTRACT,
	MULTIPLY,
	DIVIDE,
	SINE,
	COSINE,
	TANGENT,
	ATAN2,
	TANH,
	POWER, // used to implement exponent
	MIN, // NOTE: clip is implemented as a series of min and max
	MAX,
}

op_debug :: proc(graph: ^Graph) {
	op_id := len(graph.ops) - 1
	op := graph.ops[op_id]
	fmt.printf(
		"Op %v is %v %v %v\n",
		op_id,
		graph.vars_name[graph.vars[op.inputs[0]].id],
		op.kind,
		graph.vars_name[graph.vars[op.inputs[1]].id],
	)
}

/////////////////////////////////////////////////
@(private)
add_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the add operation
	assert(op.kind == OpKind.ADD)
	inputs: [2]^Var = {&graph.vars[op.inputs[0]], &graph.vars[op.inputs[1]]}
	output: ^Var = &graph.vars[op.output]
	if inputs[0].size == inputs[1].size {
		#no_bounds_check for i in 0 ..< inputs[0].size {
			output.val[i] = inputs[0].val[i] + inputs[1].val[i]
			output.dval[i] = inputs[0].dval[i] + inputs[1].dval[i]
		}
	} else {
		if inputs[0].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[1].size {
				output.val[i] = inputs[0].val[0] + inputs[1].val[i]
				output.dval[i] = inputs[0].dval[0] + inputs[1].dval[i]
			}
		}
		if inputs[1].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[0].size {
				output.val[i] = inputs[0].val[i] + inputs[1].val[0]
				output.dval[i] = inputs[0].dval[i] + inputs[1].dval[0]
			}
		}
	}
}

add_vars :: proc(x, y: Var, name: string = "_add") -> Var {
	assert(x.graph == y.graph)
	graph := x.graph

	z := var(graph, size = max(x.size, y.size), name = name)

	op: Op = {
		inputs = [2]u64{x.id, y.id},
		output = z.id,
		kind = OpKind.ADD,
	}
	append(&graph.ops, op)

	add_(graph, op)
	return z
}

add_var1 :: proc(x: f64, y: Var, name: string = "_add") -> Var {
	xvar := var_copy(y.graph, []f64{x}, fmt.aprintf("%E", x, y.graph.allocator))
	z := add_vars(xvar, y, name)
	return z
}
add_var2 :: proc(x: Var, y: f64, name: string = "_add") -> Var {
	yvar := var_copy(x.graph, []f64{y}, fmt.aprintf("%E", y, x.graph.allocator))
	z := add_vars(x, yvar, name)
	return z
}

add :: proc {
	add_vars,
	add_var1,
	add_var2,
}

/////////////////////////////////////////////////
@(private)
sub_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the subtract operation
	assert(op.kind == OpKind.SUBTRACT)
	inputs: [2]^Var = {&graph.vars[op.inputs[0]], &graph.vars[op.inputs[1]]}
	output: ^Var = &graph.vars[op.output]

	if inputs[0].size == inputs[1].size {
		#no_bounds_check for i in 0 ..< inputs[0].size {
			output.val[i] = inputs[0].val[i] - inputs[1].val[i]
			output.dval[i] = inputs[0].dval[i] - inputs[1].dval[i]
		}
	} else {
		if inputs[0].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[1].size {
				output.val[i] = inputs[0].val[0] - inputs[1].val[i]
				output.dval[i] = inputs[0].dval[0] - inputs[1].dval[i]
			}
		}
		if inputs[1].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[0].size {
				output.val[i] = inputs[0].val[i] - inputs[1].val[0]
				output.dval[i] = inputs[0].dval[i] - inputs[1].dval[0]
			}
		}
	}
}

sub_vars :: proc(x, y: Var, name: string = "_sub") -> Var {
	assert(x.graph == y.graph)
	graph := x.graph

	z := var(graph, size = max(x.size, y.size), name = name)

	op: Op = {
		inputs = [2]u64{x.id, y.id},
		output = z.id,
		kind = OpKind.SUBTRACT,
	}
	append(&graph.ops, op)

	sub_(graph, op)
	return z
}

sub_var1 :: proc(x: f64, y: Var, name: string = "_sub") -> Var {
	xvar := var_copy(y.graph, []f64{x}, fmt.aprintf("%E", x, y.graph.allocator))
	z := sub_vars(xvar, y, name)
	return z
}
sub_var2 :: proc(x: Var, y: f64, name: string = "_sub") -> Var {
	yvar := var_copy(x.graph, []f64{y}, fmt.aprintf("%E", y, x.graph.allocator))
	z := sub_vars(x, yvar, name)
	return z
}

sub :: proc {
	sub_vars,
	sub_var1,
	sub_var2,
}

/////////////////////////////////////////////////

@(private)
mult_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the MULTIPLY operation
	assert(op.kind == OpKind.MULTIPLY)
	inputs: [2]^Var = {&graph.vars[op.inputs[0]], &graph.vars[op.inputs[1]]}
	output: ^Var = &graph.vars[op.output]

	if inputs[0].size == inputs[1].size {
		#no_bounds_check for i in 0 ..< inputs[0].size {
			output.val[i] = inputs[0].val[i] * inputs[1].val[i]
			output.dval[i] = inputs[0].val[i] * inputs[1].dval[i] + inputs[0].dval[i] * inputs[1].val[i]
		}
	} else {
		if inputs[0].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[1].size {
				output.val[i] = inputs[0].val[0] * inputs[1].val[i]
				output.dval[i] = inputs[0].val[0] - inputs[1].dval[i] + inputs[0].dval[0] * inputs[1].val[i]
			}
		}
		if inputs[1].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[0].size {
				output.val[i] = inputs[0].val[i] * inputs[1].val[0]
				output.dval[i] = inputs[0].val[i] * inputs[1].dval[0] + inputs[0].dval[i] * inputs[1].val[0]
			}
		}
	}
}

mult_vars :: proc(x, y: Var, name: string = "_mul") -> Var {
	assert(x.graph == y.graph)
	graph := x.graph

	z := var(graph, size = max(x.size, y.size), name = name)

	op: Op = {
		inputs = [2]u64{x.id, y.id},
		output = z.id,
		kind = OpKind.MULTIPLY,
	}
	append(&graph.ops, op)

	mult_(graph, op)
	return z
}

mult_var1 :: proc(x: f64, y: Var, name: string = "_mult") -> Var {
	xvar := var_copy(y.graph, []f64{x}, fmt.aprintf("%E", x, y.graph.allocator))
	z := mult_vars(xvar, y, name)
	return z
}
mult_var2 :: proc(x: Var, y: f64, name: string = "_mult") -> Var {
	yvar := var_copy(x.graph, []f64{y}, fmt.aprintf("%E", y, x.graph.allocator))
	z := mult_vars(x, yvar, name)
	return z
}

mult :: proc {
	mult_vars,
	mult_var1,
	mult_var2,
}
/////////////////////////////////////////////////

@(private)
div_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the DIVIDE operation
	assert(op.kind == OpKind.DIVIDE)
	inputs: [2]^Var = {&graph.vars[op.inputs[0]], &graph.vars[op.inputs[1]]}
	output: ^Var = &graph.vars[op.output]

	// TODO pull some constants out of loops where things don't change
	if inputs[0].size == inputs[1].size {
		#no_bounds_check for i in 0 ..< inputs[0].size {
			output.val[i] = inputs[0].val[i] / inputs[1].val[i]
			output.dval[i] =
				inputs[0].dval[i] / inputs[1].val[i] -
				inputs[0].val[i] * inputs[1].dval[i] / (inputs[1].val[i] * inputs[1].val[i])
		}
	} else {
		if inputs[0].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[1].size {
				output.val[i] = inputs[0].val[0] / inputs[1].val[i]
				output.dval[i] =
					inputs[0].dval[0] / inputs[1].val[i] -
					inputs[0].val[0] * inputs[1].dval[i] / (inputs[1].val[i] * inputs[1].val[i])
			}
		}
		if inputs[1].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[0].size {
				output.val[i] = inputs[0].val[i] / inputs[1].val[0]
				output.dval[i] =
					inputs[0].dval[i] / inputs[1].val[0] -
					inputs[0].val[i] * inputs[1].dval[0] / (inputs[1].val[0] * inputs[1].val[0])
			}
		}
	}
}

div_vars :: proc(x, y: Var, name: string = "_div") -> Var {
	assert(x.graph == y.graph)
	graph := x.graph

	z := var(graph, size = max(x.size, y.size), name = name)

	op: Op = {
		inputs = [2]u64{x.id, y.id},
		output = z.id,
		kind = OpKind.DIVIDE,
	}
	append(&graph.ops, op)

	div_(graph, op)
	return z
}

div_var1 :: proc(x: f64, y: Var, name: string = "_div") -> Var {
	xvar := var_copy(y.graph, []f64{x}, fmt.aprintf("%E", x, y.graph.allocator))
	z := div_vars(xvar, y, name)
	return z
}
div_var2 :: proc(x: Var, y: f64, name: string = "_div") -> Var {
	yvar := var_copy(x.graph, []f64{y}, fmt.aprintf("%E", y, x.graph.allocator))
	z := div_vars(x, yvar, name)
	return z
}

div :: proc {
	div_vars,
	div_var1,
	div_var2,
}

/////////////////////////////////////////////////

@(private)
sin_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the SINE operation
	assert(op.kind == OpKind.SINE)
	inputs: ^Var = &graph.vars[op.inputs[0]]
	output: ^Var = &graph.vars[op.output]

	#no_bounds_check for i in 0 ..< inputs.size {
		output.val[i] = math.sin_f64(inputs.val[i])
		output.dval[i] = math.cos_f64(inputs.val[i]) * inputs.dval[i]
	}
}

sin :: proc(x: Var, name: string = "_sin") -> Var {
	graph := x.graph

	z := var(graph, size = x.size, name = name)

	op: Op = {
		inputs = [2]u64{x.id, 0},
		output = z.id,
		kind = OpKind.SINE,
	}
	append(&graph.ops, op)

	sin_(graph, op)
	return z
}

/////////////////////////////////////////////////

@(private)
cos_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the COSINE operation
	assert(op.kind == OpKind.COSINE)
	inputs: ^Var = &graph.vars[op.inputs[0]]
	output: ^Var = &graph.vars[op.output]

	#no_bounds_check for i in 0 ..< inputs.size {
		output.val[i] = math.cos_f64(inputs.val[i])
		output.dval[i] = -math.sin_f64(inputs.val[i]) * inputs.dval[i]
	}
}

cos :: proc(x: Var, name: string = "_cos") -> Var {
	graph := x.graph

	z := var(graph, size = x.size, name = name)

	op: Op = {
		inputs = [2]u64{x.id, 0},
		output = z.id,
		kind = OpKind.COSINE,
	}
	append(&graph.ops, op)

	cos_(graph, op)
	return z
}

/////////////////////////////////////////////////

@(private)
tan_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the TANGENT operation
	assert(op.kind == OpKind.TANGENT)
	inputs: ^Var = &graph.vars[op.inputs[0]]
	output: ^Var = &graph.vars[op.output]

	#no_bounds_check for i in 0 ..< inputs.size {
		output.val[i] = math.tan_f64(inputs.val[i])
		output.dval[i] = inputs.dval[i] / math.pow_f64(math.cos_f64(inputs.val[i]), 2.0)
	}
}

tan :: proc(x: Var, name: string = "_tan") -> Var {
	graph := x.graph

	z := var(graph, size = x.size, name = name)

	op: Op = {
		inputs = [2]u64{x.id, 0},
		output = z.id,
		kind = OpKind.TANGENT,
	}
	append(&graph.ops, op)

	tan_(graph, op)
	return z
}

/////////////////////////////////////////////////

@(private)
tanh_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the TANH operation
	assert(op.kind == OpKind.TANH)
	inputs: ^Var = &graph.vars[op.inputs[0]]
	output: ^Var = &graph.vars[op.output]

	#no_bounds_check for i in 0 ..< inputs.size {
		output.val[i] = math.tanh(inputs.val[i])
		output.dval[i] = inputs.dval[i] / math.pow_f64(math.cosh(inputs.val[i]), 2.0)
	}
}

tanh :: proc(x: Var, name: string = "_tanh") -> Var {
	graph := x.graph

	z := var(graph, size = x.size, name = name)

	op: Op = {
		inputs = [2]u64{x.id, 0},
		output = z.id,
		kind = OpKind.TANH,
	}
	append(&graph.ops, op)

	tanh_(graph, op)
	return z
}


/////////////////////////////////////////////////

@(private)
atan2_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the ATAN2 operation
	assert(op.kind == OpKind.ATAN2)
	in_y: ^Var = &graph.vars[op.inputs[0]]
	in_x: ^Var = &graph.vars[op.inputs[1]]
	output: ^Var = &graph.vars[op.output]

	// TODO pull some constants out of loops where things don't change
	if in_y.size == in_x.size {
		#no_bounds_check for i in 0 ..< in_y.size {
			output.val[i] = math.atan2_f64(in_y.val[i], in_x.val[i])
			denom: f64 = 1.0 / (math.pow_f64(in_x.val[i], 2.0) + math.pow_f64(in_y.val[i], 2.0))
			output.dval[i] = (in_y.dval[i] * in_x.val[i] - in_x.dval[i] * in_y.val[i]) * denom
		}
	} else {
		if in_y.size == 1 {
			#no_bounds_check for i in 0 ..< in_x.size {
				output.val[i] = math.atan2_f64(in_y.val[0], in_x.val[i])
				denom: f64 = 1.0 / (math.pow_f64(in_x.val[i], 2.0) + math.pow_f64(in_y.val[0], 2.0))
				output.dval[i] = (in_y.dval[0] * in_x.val[i] - in_x.dval[i] * in_y.val[0]) * denom
			}
		}
		if in_x.size == 1 {
			#no_bounds_check for i in 0 ..< in_y.size {
				output.val[i] = math.atan2_f64(in_y.val[i], in_x.val[0])
				denom: f64 = 1.0 / (math.pow_f64(in_x.val[0], 2.0) + math.pow_f64(in_y.val[i], 2.0))
				output.dval[i] = (in_y.dval[i] * in_x.val[0] - in_x.dval[0] * in_y.val[i]) * denom
			}
		}
	}
}

atan2_vars :: proc(y, x: Var, name: string = "_atan2") -> Var {
	assert(x.graph == y.graph)
	graph := x.graph

	z := var(graph, size = max(x.size, y.size), name = name)

	op: Op = {
		inputs = [2]u64{y.id, x.id},
		output = z.id,
		kind = OpKind.ATAN2,
	}
	append(&graph.ops, op)

	atan2_(graph, op)
	return z
}

atan2_var1 :: proc(y: f64, x: Var, name: string = "_atan2") -> Var {
	yvar := var_copy(x.graph, []f64{y}, fmt.aprintf("%E", y, x.graph.allocator))
	z := atan2_vars(yvar, x, name)
	return z
}

atan2_var2 :: proc(y: Var, x: f64, name: string = "_atan2") -> Var {
	xvar := var_copy(y.graph, []f64{x}, fmt.aprintf("%E", x, y.graph.allocator))
	z := atan2_vars(y, xvar, name)
	return z
}

atan2 :: proc {
	atan2_vars,
	atan2_var1,
	atan2_var2,
}

/////////////////////////////////////////////////

@(private)
power_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the POWER operation
	assert(op.kind == OpKind.POWER)
	inputs: [2]^Var = {&graph.vars[op.inputs[0]], &graph.vars[op.inputs[1]]}
	output: ^Var = &graph.vars[op.output]

	// TODO pull some constants out of loops where things don't change?
	// BUG math.ln_f64 can't handle values < 0!
	if inputs[0].size == inputs[1].size {
		#no_bounds_check for i in 0 ..< inputs[0].size {
			output.val[i] = math.pow_f64(inputs[0].val[i], inputs[1].val[i])
			output.dval[i] =
				inputs[1].val[i] * math.pow_f64(inputs[0].val[i], inputs[1].val[i] - 1.0) * inputs[0].dval[i]
			if inputs[1].dval[i] != 0 {
				assert(inputs[0].val[i] > 0)
				output.dval[i] +=
					math.pow_f64(inputs[0].val[i], inputs[1].val[i]) *
					math.ln_f64(inputs[0].val[i]) *
					inputs[1].dval[i]
			}
		}
	} else {
		if inputs[0].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[1].size {
				output.val[i] = math.pow_f64(inputs[0].val[0], inputs[1].val[i])
				output.dval[i] =
					inputs[1].val[i] * math.pow_f64(inputs[0].val[0], inputs[1].val[i] - 1.0) * inputs[0].dval[0]
				if inputs[1].dval[i] != 0 {
					assert(inputs[0].val[0] > 0)
					output.dval[i] +=
						math.pow_f64(inputs[0].val[0], inputs[1].val[i]) *
						math.ln_f64(inputs[0].val[0]) *
						inputs[1].dval[i]
				}
			}
		}
		if inputs[1].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[0].size {
				output.val[i] = math.pow_f64(inputs[0].val[i], inputs[1].val[0])
				output.dval[i] =
					inputs[1].val[0] * math.pow_f64(inputs[0].val[i], inputs[1].val[0] - 1.0) * inputs[0].dval[i]
				if inputs[1].dval[0] != 0 {
					assert(inputs[0].val[i] > 0)
					output.dval[i] =
						math.pow_f64(inputs[0].val[i], inputs[1].val[0]) *
						math.ln_f64(inputs[0].val[i]) *
						inputs[1].dval[0]
				}
			}
		}
	}
}

power_vars :: proc(x, power: Var, name: string = "_pow") -> Var {
	assert(x.graph == power.graph)
	graph := x.graph

	z := var(graph, size = max(x.size, power.size), name = name)

	op: Op = {
		inputs = [2]u64{x.id, power.id},
		output = z.id,
		kind = OpKind.POWER,
	}
	append(&graph.ops, op)

	power_(graph, op)
	return z
}

power_var1 :: proc(x: f64, y: Var, name: string = "_power") -> Var {
	xvar := var_copy(y.graph, []f64{x}, fmt.aprintf("%E", x, y.graph.allocator))
	z := power_vars(xvar, y, name)
	return z
}
power_var2 :: proc(x: Var, y: f64, name: string = "_power") -> Var {
	yvar := var_copy(x.graph, []f64{y}, fmt.aprintf("%E", y, x.graph.allocator))
	z := power_vars(x, yvar, name)
	return z
}

power :: proc {
	power_vars,
	power_var1,
	power_var2,
}

exp :: proc(x: Var, name: string = "_exp") -> Var {
	e := var_copy(x.graph, []f64{math.E}, "math.E")
	z := power_vars(e, x, name)
	return z
}


/////////////////////////////////////////////////
// called vmin to not conflict with builtin min

@(private)
vmin_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the MIN operation
	assert(op.kind == OpKind.MIN)
	inputs: [2]^Var = {&graph.vars[op.inputs[0]], &graph.vars[op.inputs[1]]}
	output: ^Var = &graph.vars[op.output]

	if inputs[0].size == inputs[1].size {
		#no_bounds_check for i in 0 ..< inputs[0].size {
			if inputs[0].val[i] > inputs[1].val[i] {
				output.val[i] = inputs[1].val[i]
				output.dval[i] = inputs[1].dval[i]
			} else {
				output.val[i] = inputs[0].val[i]
				output.dval[i] = inputs[0].dval[i]
			}
		}
	} else {
		if inputs[0].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[1].size {
				if inputs[0].val[0] > inputs[1].val[i] {
					output.val[i] = inputs[1].val[i]
					output.dval[i] = inputs[1].dval[i]
				} else {
					output.val[i] = inputs[0].val[0]
					output.dval[i] = inputs[0].dval[0]
				}
			}
		}
		if inputs[1].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[0].size {
				if inputs[0].val[i] > inputs[1].val[0] {
					output.val[i] = inputs[1].val[0]
					output.dval[i] = inputs[1].dval[0]
				} else {
					output.val[i] = inputs[0].val[i]
					output.dval[i] = inputs[0].dval[i]
				}
			}
		}
	}
}

vmin_vars :: proc(x, y: Var, name: string = "_min") -> Var {
	assert(x.graph == y.graph)
	graph := x.graph

	z := var(graph, size = max(x.size, y.size), name = name)

	op: Op = {
		inputs = [2]u64{x.id, y.id},
		output = z.id,
		kind = OpKind.MIN,
	}
	append(&graph.ops, op)

	vmin_(graph, op)
	return z
}


vmin_var1 :: proc(x: f64, y: Var, name: string = "_min") -> Var {
	xvar := var_copy(y.graph, []f64{x}, fmt.aprintf("%E", x, y.graph.allocator))
	z := vmin_vars(xvar, y, name)
	return z
}
vmin_var2 :: proc(x: Var, y: f64, name: string = "_min") -> Var {
	yvar := var_copy(x.graph, []f64{y}, fmt.aprintf("%E", y, x.graph.allocator))
	z := vmin_vars(x, yvar, name)
	return z
}

vmin :: proc {
	vmin_vars,
	vmin_var1,
	vmin_var2,
}

/////////////////////////////////////////////////
// called vmax to not conflict with bulitin max

@(private)
vmax_ :: proc(graph: ^Graph, op: Op) {
	// Actually perform the MAX operation
	assert(op.kind == OpKind.MAX)
	inputs: [2]^Var = {&graph.vars[op.inputs[0]], &graph.vars[op.inputs[1]]}
	output: ^Var = &graph.vars[op.output]

	if inputs[0].size == inputs[1].size {
		#no_bounds_check for i in 0 ..< inputs[0].size {
			if inputs[0].val[i] < inputs[1].val[i] {
				output.val[i] = inputs[1].val[i]
				output.dval[i] = inputs[1].dval[i]
			} else {
				output.val[i] = inputs[0].val[i]
				output.dval[i] = inputs[0].dval[i]
			}
		}
	} else {
		if inputs[0].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[1].size {
				if inputs[0].val[0] < inputs[1].val[i] {
					output.val[i] = inputs[1].val[i]
					output.dval[i] = inputs[1].dval[i]
				} else {
					output.val[i] = inputs[0].val[0]
					output.dval[i] = inputs[0].dval[0]
				}
			}
		}
		if inputs[1].size == 1 {
			#no_bounds_check for i in 0 ..< inputs[0].size {
				if inputs[0].val[i] < inputs[1].val[0] {
					output.val[i] = inputs[1].val[0]
					output.dval[i] = inputs[1].dval[0]
				} else {
					output.val[i] = inputs[0].val[i]
					output.dval[i] = inputs[0].dval[i]
				}
			}
		}
	}
}

// max(x, 0) is equivalent to ReLU(x)
vmax_vars :: proc(x, y: Var, name: string = "_max") -> Var {
	assert(x.graph == y.graph)
	graph := x.graph

	z := var(graph, size = max(x.size, y.size), name = name)

	op: Op = {
		inputs = [2]u64{x.id, y.id},
		output = z.id,
		kind = OpKind.MAX,
	}
	append(&graph.ops, op)

	vmax_(graph, op)
	return z
}

vmax_var1 :: proc(x: f64, y: Var, name: string = "_max") -> Var {
	xvar := var_copy(y.graph, []f64{x}, fmt.aprintf("%E", x, y.graph.allocator))
	z := vmax_vars(xvar, y, name)
	return z
}
vmax_var2 :: proc(x: Var, y: f64, name: string = "_max") -> Var {
	yvar := var_copy(x.graph, []f64{y}, fmt.aprintf("%E", y, x.graph.allocator))
	z := vmax_vars(x, yvar, name)
	return z
}

vmax :: proc {
	vmax_vars,
	vmax_var1,
	vmax_var2,
}

/////////////////////////////////////////////////

clip_vars :: proc(x, low, high: Var, name: string = "_clip") -> Var {
	// stack min and max
	z1 := vmin(x, high, strings.concatenate({name, "_high"}, allocator = x.graph.allocator))
	z2 := vmax(z1, low, strings.concatenate({name, "_low"}, allocator = x.graph.allocator))
	return z2
}

clip_var1 :: proc(x: f64, low, high: Var, name: string = "_clip") -> Var {
	xvar := var_copy(low.graph, []f64{x}, fmt.aprintf("%E", x, low.graph.allocator))
	z := clip_vars(xvar, low, high, name)
	return z
}
clip_var2 :: proc(x: Var, low: f64, high: f64, name: string = "_clip") -> Var {
	low_var := var_copy(x.graph, []f64{low}, fmt.aprintf("%E", low, x.graph.allocator))
	high_var := var_copy(x.graph, []f64{high}, fmt.aprintf("%E", high, x.graph.allocator))
	z := clip_vars(x, low_var, high_var, name)
	return z
}

clip :: proc {
	clip_vars,
	clip_var1,
	clip_var2,
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Testing!

@(test)
test_add :: proc(t: ^testing.T) {
	graph := graph_init()

	x: Var = var(graph, val = []f64{4}, name = "x")
	y: Var = var(graph, val = []f64{7}, name = "y")
	z: Var = add(x, y)

	{
		uncertainty(graph, x)
		testing.expect_value(t, graph.vars[z.id].val[0], 11)
		testing.expect_value(t, graph.vars[z.id].dval[0], 1.0)
	}
	{
		uncertainty(graph, y)
		testing.expect_value(t, graph.vars[z.id].val[0], 11)
		testing.expect_value(t, graph.vars[z.id].dval[0], 1.0)
	}
}


@(test)
test_subtract :: proc(t: ^testing.T) {
	graph := graph_init()
	x: Var = var(graph, val = []f64{4, 5}, name = "x")
	y: Var = var(graph, val = []f64{7, 1}, name = "y")

	z: Var = sub(x, y)

	{
		uncertainty(graph, x)
		testing.expect_value(t, graph.vars[z.id].val[0], -3)
		testing.expect_value(t, graph.vars[z.id].dval[0], 1.0)
	}
	{
		uncertainty(graph, y)
		testing.expect_value(t, graph.vars[z.id].val[0], -3)
		testing.expect_value(t, graph.vars[z.id].dval[0], -1.0)
	}
}


@(test)
test_multiply :: proc(t: ^testing.T) {
	graph := graph_init()
	x: Var = var(graph, val = []f64{4, 5}, name = "x")
	y: Var = var(graph, val = []f64{7}, name = "y")

	z: Var = mult(x, y)

	{
		uncertainty(graph, x)
		testing.expect_value(t, graph.vars[z.id].val[0], 28)
		testing.expect_value(t, graph.vars[z.id].dval[0], 7.0)
	}
	{
		uncertainty(graph, y)
		testing.expect_value(t, graph.vars[z.id].val[0], 28)
		testing.expect_value(t, graph.vars[z.id].dval[0], 4.0)
	}
}


@(test)
test_limits :: proc(t: ^testing.T) {
	graph := graph_init()

	x: Var = var(graph, val = []f64{3}, name = "x")
	y: Var = var(graph, val = []f64{4}, name = "y")

	zmin: Var = vmin(x, y)
	zmax: Var = vmax(x, y)

	testing.expect_value(t, graph.vars[zmin.id].val[0], 3)
	testing.expect_value(t, graph.vars[zmax.id].val[0], 4)

	uncertainty(graph, x)
	testing.expect_value(t, graph.vars[zmin.id].dval[0], 1.0)
	testing.expect_value(t, graph.vars[zmax.id].dval[0], 0.0)

	uncertainty(graph, y)
	testing.expect_value(t, graph.vars[zmin.id].dval[0], 0.0)
	testing.expect_value(t, graph.vars[zmax.id].dval[0], 1.0)
}


@(test)
test_series1 :: proc(t: ^testing.T) {
	graph := graph_init()

	x: Var = var(graph, val = []f64{0.9}, name = "x")
	w: Var = var(graph, val = []f64{1.2}, name = "w")
	a: Var = var(graph, val = []f64{1.4}, name = "a")

	w1 := mult(x, w)
	w2 := sin(w1)
	z := mult(a, w2)

	uncertainty(graph, a)
	testing.expect_value(t, graph.vars[z.id].dval[0], math.sin_f64(w.val[0] * x.val[0]))

	uncertainty(graph, w)
	testing.expect_value(t, graph.vars[z.id].dval[0], a.val[0] * x.val[0] * math.cos_f64(w.val[0] * x.val[0]))

	uncertainty(graph, x)
	testing.expect_value(t, graph.vars[z.id].dval[0], a.val[0] * w.val[0] * math.cos_f64(w.val[0] * x.val[0]))
}


@(test)
test_branching :: proc(t: ^testing.T) {
	graph := graph_init()
	x: Var = var(graph, val = []f64{1.0}, name = "x")
	y: Var = var(graph, val = []f64{2.0}, name = "y")
	A: Var = var(graph, val = []f64{3.0}, name = "A")

	w1 := add(x, y)
	w2 := mult(x, w1)

	w3 := mult(w2, A)
	w4 := sub(w2, A)

	z1 := sin(w3) // sin(x*(x+y)*A)
	z2 := cos(w4) // cos(x*(x+y) - A)

	{
		uncertainty(graph, x)
		testing.expect_value(
			t,
			graph.vars[z1.id].dval[0],
			A.val[0] * (2.0 * x.val[0] + y.val[0]) * math.cos_f64(A.val[0] * x.val[0] * (x.val[0] + y.val[0])),
		)

		testing.expect_value(
			t,
			graph.vars[z2.id].dval[0],
			(2.0 * x.val[0] + y.val[0]) * math.sin_f64(A.val[0] - x.val[0] * (x.val[0] + y.val[0])),
		)


	}

	{
		uncertainty(graph, y)
		testing.expect_value(
			t,
			graph.vars[z1.id].dval[0],
			A.val[0] * x.val[0] * math.cos_f64(A.val[0] * x.val[0] * (x.val[0] + y.val[0])),
		)
		testing.expect_value(
			t,
			graph.vars[z2.id].dval[0],
			x.val[0] * math.sin_f64(A.val[0] - x.val[0] * (x.val[0] + y.val[0])),
		)
	}
}


@(test)
test_optional_constants :: proc(t: ^testing.T) {
	graph := graph_init()

	x := var_copy(graph, []f64{32, 48}, "x")

	y := add(x, 1)
	testing.expect_value(t, y.val[0], 33)
	testing.expect_value(t, y.val[1], 49)

	z := add(2, x)
	testing.expect_value(t, z.val[0], 34)
	testing.expect_value(t, z.val[1], 50)
}


@(test)
test_iteration :: proc(t: ^testing.T) {
	// Numerically find the minimum of a parabola Ax^2 + Bx + C
	graph := graph_init()

	x := var_copy(graph, []f64{2}, "x")
	a := var_copy(graph, []f64{1}, "a")
	b := var_copy(graph, []f64{4}, "b")
	c := var_copy(graph, []f64{2}, "c")

	k1 := mult(power(x, 2), a)
	k2 := mult(x, b)
	y := add(add(k1, k2), c, name = "y")

	ALPHA :: 0.5
	for _ in 0 ..< 20 {
		uncertainty_collect(graph, targets = []Var{y}, knobs = []Var{x})
		graph.vars[x.id].val[0] -= ALPHA * graph.vars[y.id].dval[0]
	}

	testing.expect(t, math.abs(graph.vars[y.id].val[0] - (-2.0)) < 0.005)
}

@(test)
test_readme :: proc(t: ^testing.T) {
	graph: ^Graph = graph_init()
	x: Var = var_copy(graph, []f64{3}, "x")
	y: Var = var_copy(graph, []f64{4}, "y")

	z: Var = mult(x, y, "z")

	uncertainty_collect(graph, targets = []Var{z}, knobs = []Var{x, y})
}

@(test)
test_exp :: proc(t: ^testing.T) {
	graph := graph_init()
	x := var_copy(graph, []f64{1, 2, 3, 4}, "x")
	y := exp(x, "y")

	uncertainty(graph, x)

	for i in 0 ..< x.size {
		testing.expect(t, math.abs(graph.vars[y.id].dval[i] - math.exp_f64(graph.vars[x.id].val[i])) < 1e-6)
	}
}

@(test)
test_tan :: proc(t: ^testing.T) {
	graph := graph_init()

	x := var_copy(graph, []f64{-0.5, 0.0, 1.0}, "x")

	y := tan(x, "y")
	uncertainty(graph, x)

	testing.expect(t, math.abs(graph.vars[y.id].val[0] - (-0.5463024)) < 1e-6)
	testing.expect(t, math.abs(graph.vars[y.id].val[1] - (0)) < 1e-6)
	testing.expect(t, math.abs(graph.vars[y.id].val[2] - (1.5574077)) < 1e-6)

	testing.expect(t, math.abs(graph.vars[y.id].dval[0] - (1.2984464)) < 1e-6)
	testing.expect(t, math.abs(graph.vars[y.id].dval[1] - (1)) < 1e-6)
	testing.expect(t, math.abs(graph.vars[y.id].dval[2] - (3.4255188)) < 1e-6)
}

@(test)
test_tanh :: proc(t: ^testing.T) {
	graph := graph_init()
	x := var_copy(graph, []f64{-0.5, 0.0, 1.0}, "x")

	y := tanh(x, "y")
	uncertainty(graph, x)

	graph_print(graph)
	testing.expect(t, math.abs(graph.vars[y.id].val[0] - (-0.4621171)) < 1e-6)
	testing.expect(t, math.abs(graph.vars[y.id].val[1] - (0)) < 1e-6)
	testing.expect(t, math.abs(graph.vars[y.id].val[2] - (0.7615941)) < 1e-6)

	testing.expect(t, math.abs(graph.vars[y.id].dval[0] - (0.7864477)) < 1e-6)
	testing.expect(t, math.abs(graph.vars[y.id].dval[1] - (1)) < 1e-6)
	testing.expect(t, math.abs(graph.vars[y.id].dval[2] - (0.4199743)) < 1e-6)
}

@(test)
test_atan2 :: proc(t: ^testing.T) {
	graph := graph_init()
	x := var_copy(graph, []f64{-0.5, 0.0, 1.2, 0.9, 1.1, 0.0, -0.8, -1.4}, "x")
	y := var_copy(graph, []f64{1.2, 0.9, 0.7, 0.0, -0.6, -1.1, -1.2, 0.0}, "y")

	z := atan2(y, x, "z")

	{
		// Check basic results of atan2 in all four quadrants
		z_answers := []f64{1.9655874, 1.5707963, 0.5280744, 0.0, -0.4993467, -1.5707963, -2.1587989, 3.1415926}
		assert(z.size == u64(len(z_answers)))

		end := len(z_answers) - 1
		z_answers[end] *= math.sign(graph.vars[z.id].val[end]) // don't care about matching sign of pi

		for i in 0 ..< len(z_answers) {
			testing.expect(t, math.abs(graph.vars[z.id].val[i] - z_answers[i]) < 1e-6, fmt.tprintf("z error\n"))
		}
	}

	{
		uncertainty(graph, x)
		dz_answers := []f64{-0.7100591, -1.1111111, -0.3626943, 0.0, 0.3821656, 0.9090909, 0.5769230, 0.0}
		assert(z.size == u64(len(dz_answers)))

		for i in 0 ..< len(dz_answers) {
			testing.expect(t, math.abs(graph.vars[z.id].dval[i] - dz_answers[i]) < 1e-6, fmt.tprintf("dz/dx error\n"))
		}
	}

	{
		uncertainty(graph, y)
		dz_answers := []f64{-0.29585799, 0.0, 0.62176166, 1.11111111, 0.70063694, 0.0, -0.38461538, -0.71428571}
		assert(z.size == u64(len(dz_answers)))

		for i in 0 ..< len(dz_answers) {
			testing.expect(t, math.abs(graph.vars[z.id].dval[i] - dz_answers[i]) < 1e-6, fmt.tprintf("dz/dy error\n"))
		}
	}
}

