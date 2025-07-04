// =====================================================================
// Enhanced Instruction Memory Module
// Features:
// - Stores 16 instructions (expandable to more if needed)
// - Uses a 4-bit program counter for addressing
// - Outputs the full instr_opcode for CPU execution
// - Optional reset capability with NOP instruction at PC=0
// =====================================================================

module instruction_ram(
input wire clk,             // Clock signal (for synchronous read if needed)
input wire reset,           // Reset signal
input wire [3:0] pc_in,     // 4-bit Program Counter input
output wire [15:0] instr_opcode // 16-bit instruction output
);

// Instruction Memory (16 instructions of 16 bits each)
reg [15:0] imem [0:15];

integer i; // Moved loop variable declaration to module scope

initial begin
    // Initialize instruction memory with a sample program
    // This program is for demonstration. Replace with actual program.
    // Assume opcode format: [opcode (4)|rd (3)|rs1 (3)|rs2 (3)|imm (3)] for R-type/I-type
    // Or [opcode (4)|branch_target (12)] for J-type
    // Or [opcode (4)|rs1 (3)|imm (9)] for Load/Store etc.

    for (i = 0; i < 16; i = i + 1) begin // Corrected loop variable declaration
        imem[i] = 16'h0000; // Initialize all to NOP or 0
    end
    imem[0] = 16'h1234; // ADD R1, R2, R3 (opcode 1, rd=1, rs1=2, rs2=3) - Placeholder
    imem[1] = 16'h2452; // ADDI R4, R5, #2 (opcode 2, rd=4, rs1=5, imm=2) - Placeholder
    imem[2] = 16'h3678; // SUB R6, R7, R8 - Placeholder
    imem[3] = 16'h4891; // LD R8, (R9 + #1) - Placeholder
    imem[4] = 16'h5ABA; // ST R10, (R11 + #10) - Placeholder
    imem[5] = 16'h6CDE; // XOR R12, R13, R14 - Placeholder
    imem[6] = 16'h7F01; // BEQ R15, R0, +1 (branch if R15 == R0, to PC+1) - Placeholder
    imem[7] = 16'h8002; // JUMP PC+2 (unconditional jump) - Placeholder
    imem[8] = 16'h9123; // NOP - Placeholder
    imem[9] = 16'h0000; // NOP - Placeholder
    imem[10] = 16'h0000; // NOP - Placeholder
    imem[11] = 16'h0000; // NOP - Placeholder
    imem[12] = 16'h0000; // NOP - Placeholder
    imem[13] = 16'h0000; // NOP - Placeholder
    imem[14] = 16'h0000; // NOP - Placeholder
    imem[15] = 16'h0000; // NOP - Placeholder
end

// Instruction fetch logic
assign instr_opcode = imem[pc_in];


endmodule

// ===============================================================================
// Branch Target Buffer (BTB) Module
// Features:
// - Stores predicted next PC for branches.
// - Improves pipeline performance by reducing branch prediction penalty.
// - Updates on misprediction.
// ===============================================================================
module branch_target_buffer(
input wire clk,
input wire reset,
input wire [3:0] pc_in,             // Current PC to check for prediction
input wire [3:0] branch_resolved_pc, // PC of branch instruction whose outcome is resolved
input wire branch_resolved_pc_valid, // Indicates if branch_resolved_pc is valid
input wire [3:0] branch_resolved_target_pc, // Actual target PC of the resolved branch
input wire branch_resolved_taken, // Actual outcome of the resolved branch (taken/not taken)

output wire [3:0] predicted_next_pc, // Predicted next PC
output wire predicted_taken         // Predicted branch outcome (taken/not taken)



);


// Simple BTB: Stores target PC for each instruction address
// Each entry: {predicted_taken_bit, predicted_target_pc[3:0]}
reg [4:0] btb_table [0:15]; // 16 entries, 5 bits each (1 for taken, 4 for PC)

integer i; // Moved loop variable declaration to module scope

initial begin
    // Initialize BTB (e.g., all not taken, target PC is 0)
    for (i = 0; i < 16; i = i + 1) begin // Corrected loop variable declaration
        btb_table[i] = 5'b0_0000;
    end
end

// Prediction logic (combinational read)
assign predicted_next_pc = btb_table[pc_in][3:0];
assign predicted_taken = btb_table[pc_in][4];

// Update logic (synchronous write)
always @(posedge clk or posedge reset) begin
    if (reset) begin
        for (i = 0; i < 16; i = i + 1) begin // Corrected loop variable declaration
            btb_table[i] = 5'b0_0000;
        end
    end else begin
        if (branch_resolved_pc_valid) begin
            // Update BTB entry for the resolved branch
            btb_table[branch_resolved_pc] <= {branch_resolved_taken, branch_resolved_target_pc};
        end
    end
end



endmodule

// =====================================================================
// Register File Module
// Features:
// - 8 4-bit registers (R0-R7)
// - R0 is hardwired to 0
// - Dual read ports for simultaneous operand fetching
// - Single write port for result write-back
// =====================================================================
module register_file(
input wire clk,             // Clock signal for synchronous write
input wire reset,           // Reset signal
input wire regfile_write_enable, // Enable signal for write operation
input wire [2:0] write_addr, // 3-bit address for write operation
input wire [3:0] write_data, // 4-bit data to write


input wire [2:0] read_addr1, // 3-bit address for read port 1
input wire [2:0] read_addr2, // 3-bit address for read port 2
output wire [3:0] read_data1, // 4-bit data from read port 1
output wire [3:0] read_data2  // 4-bit data from read port 2



);

// 8 registers, each 4 bits wide
reg [3:0] registers [0:7];

integer i; // Moved loop variable declaration to module scope

initial begin
    // Initialize all registers to 0 on startup
    for (i = 0; i < 8; i = i + 1) begin // Corrected loop variable declaration
        registers[i] = 4'h0;
    end
end

// Write operation (synchronous)
always @(posedge clk or posedge reset) begin
    if (reset) begin
        for (i = 0; i < 8; i = i + 1) begin // Corrected loop variable declaration
            registers[i] = 4'h0;
        end
    end else if (regfile_write_enable) begin
        // R0 is hardwired to 0, so never write to it
        if (write_addr != 3'b000) begin
            registers[write_addr] <= write_data;
        end
    end
end

// Read operations (combinational)
assign read_data1 = (read_addr1 == 3'b000) ? 4'h0 : registers[read_addr1]; // R0 always reads 0
assign read_data2 = (read_addr2 == 3'b000) ? 4'h0 : registers[read_addr2]; // R0 always reads 0



endmodule

// =====================================================================
// ALU Module (Arithmetic Logic Unit)
// Features:
// - Performs basic arithmetic and logical operations.
// - Outputs 4-bit result and 4 flags (Zero, Negative, Carry, Overflow).
// =====================================================================
module alu_unit(
input wire [3:0] alu_operand1, // First 4-bit operand
input wire [3:0] alu_operand2, // Second 4-bit operand
input wire [2:0] alu_op,       // 3-bit ALU operation code
// 3'b000: ADD
// 3'b001: SUB
// 3'b010: AND
// 3'b011: OR
// 3'b100: XOR
// 3'b101: SLT (Set Less Than)
// Other codes can be defined for shifts, etc.
output reg [3:0] alu_result,   // 4-bit result
output reg zero_flag,          // Result is zero
output reg negative_flag,      // Result is negative (MSB is 1)
output reg carry_flag,         // Carry out from addition or borrow from subtraction
output reg overflow_flag       // Signed overflow
);


always @(*) begin
    alu_result = 4'h0;
    zero_flag = 1'b0;
    negative_flag = 1'b0;
    carry_flag = 1'b0;
    overflow_flag = 1'b0;

    case (alu_op)
        3'b000: begin // ADD
            alu_result = alu_operand1 + alu_operand2;
            carry_flag = (alu_operand1 + alu_operand2) > 4'b1111; // Check for unsigned carry out
            overflow_flag = ((!alu_operand1[3] && !alu_operand2[3] && alu_result[3]) || (alu_operand1[3] && alu_operand2[3] && !alu_result[3])); // Signed overflow
        end
        3'b001: begin // SUB (using 2's complement addition)
            alu_result = alu_operand1 - alu_operand2;
            carry_flag = (alu_operand1 >= alu_operand2); // For subtraction, carry_flag usually means no borrow
            overflow_flag = ((alu_operand1[3] && !alu_operand2[3] && !alu_result[3]) || (!alu_operand1[3] && alu_operand2[3] && alu_result[3])); // Signed overflow
        end
        3'b010: begin // AND
            alu_result = alu_operand1 & alu_operand2;
        end
        3'b011: begin // OR
            alu_result = alu_operand1 | alu_operand2;
        end
        3'b100: begin // XOR
            alu_result = alu_operand1 ^ alu_operand2;
        end
        3'b101: begin // SLT (Set Less Than)
            alu_result = ($signed(alu_operand1) < $signed(alu_operand2)) ? 4'h1 : 4'h0;
        end
        default: begin
            alu_result = 4'h0; // NOP or undefined
        end
    endcase

    // Common flag calculations
    if (alu_result == 4'h0)
        zero_flag = 1'b1;
    if (alu_result[3] == 1'b1) // Check MSB for signed negative
        negative_flag = 1'b1;
end



endmodule

// =====================================================================
// Data Memory Module
// Features:
// - Simple synchronous read, asynchronous write data memory
// - Can be expanded to different sizes or types
// =====================================================================
module data_mem(
input wire clk,             // Clock signal for synchronous operation
input wire mem_write_enable, // Write enable signal
input wire mem_read_enable,  // Read enable signal (for synchronous read)
input wire [3:0] addr,       // 4-bit address input
input wire [3:0] write_data, // 4-bit data to write
output reg [3:0] read_data   // 4-bit data read
);


reg [3:0] dmem [0:15]; // 16 entries, 4 bits each

integer i; // Moved loop variable declaration to module scope

initial begin
    // Initialize data memory
    for (i = 0; i < 16; i = i + 1) begin // Corrected loop variable declaration
        dmem[i] = 4'h0;
    end
end

// Write operation (synchronous)
always @(posedge clk) begin // Corrected: Used 'begin' here
    if (mem_write_enable) begin
        dmem[addr] <= write_data;
    end
end

// Read operation (synchronous, value is stable on next clock cycle)
always @(posedge clk) begin // Corrected: Used 'begin' here
    if (mem_read_enable) begin
        read_data <= dmem[addr];
    end
end



endmodule

// ======================================================================
// Quantum Entropy Detector Module (Simplified Placeholder)
// Features:
// - Simulates a very basic "quantum entropy" or "chaos" level.
// - This is a conceptual module; a real one would involve complex quantum state measurements.
// - Output `entropy_value` represents disorder or uncertainty.
// ======================================================================
module quantum_entropy_detector(
input wire clk,
input wire reset,
input wire [3:0] instr_opcode, // Example: Opcode can influence entropy (from IF/ID)
input wire [3:0] alu_result,   // Example: ALU result can influence entropy (from EX/MEM)
input wire zero_flag,          // Example: ALU flags can influence entropy (from EX/MEM)
// ... other internal CPU signals that could affect quantum state ...
output reg [7:0] entropy_score_out // CHANGED to 8-bit to match fsm_entropy_overlay
);


// Placeholder: Entropy value increases with complex/branching instructions
// and decreases with NOPs or simple operations.
// In a real Archon-like system, this would be derived from actual quantum
// measurements or a complex internal quantum state model.
always @(posedge clk or posedge reset) begin
    if (reset) begin
        entropy_score_out <= 8'h00;
    end else begin
        // Simple heuristic: increase entropy on non-NOP, non-trivial ALU ops
        // and based on how 'unexpected' an ALU result might be.
        // Using 4 MSBs of 16-bit instr_opcode as actual opcode
        if (instr_opcode != 4'h9) begin // If not a NOP (assuming 4'h9 is NOP opcode)
            if (alu_result == 4'h0 && !zero_flag) begin // An "unexpected" zero result (not explicitly set)
                entropy_score_out <= entropy_score_out + 8'h10; // Larger jump for anomaly
            end else if (entropy_score_out < 8'hFF) begin // Prevent overflow
                entropy_score_out <= entropy_score_out + 8'h01;
            end
        end else begin
            // Reduce entropy during NOPs or idle cycles
            if (entropy_score_out > 8'h00)
                entropy_score_out <= entropy_score_out - 8'h01;
        end
    end
end



endmodule

// ======================================================================
// Chaos Detector Module (Simplified Placeholder)
// Features:
// - Simulates a rising "chaos score" based on unexpected events.
// - This is a conceptual module, representing system instability.
// ======================================================================
module chaos_detector(
input wire clk,
input wire reset,
input wire branch_mispredicted, // Example: Branch misprediction contributes to chaos (from MEM/WB)
input wire [3:0] mem_access_addr, // Example: Erratic memory access patterns (from MEM)
input wire [3:0] data_mem_read_data, // Example: Unexpected data values (from MEM)

output reg [15:0] chaos_score_out // 16-bit output


);

// Placeholder: Chaos score increases with mispredictions and erratic behavior.
// In a real system, this would be from complex monitoring.
always @(posedge clk or posedge reset) begin
    if (reset) begin
        chaos_score_out <= 16'h0000;
    end else begin
        if (branch_mispredicted) begin
            chaos_score_out <= chaos_score_out + 16'h0100; // Significant jump for misprediction
        end

        // Simulate some "erratic" memory access contributing to chaos
        // This is purely illustrative and would need robust detection logic
        // Example: Accessing a forbidden address or unusual data for an address
        if (mem_access_addr == 4'hF && data_mem_read_data == 4'h5) begin // Specific "bad" read pattern
            chaos_score_out <= chaos_score_out + 16'h0050;
        end

        // Gradually decay chaos over time if no new events
        if (chaos_score_out > 16'h0000) begin
            chaos_score_out <= chaos_score_out - 16'h0001;
        end
    end
end



endmodule

// ======================================================================
// Pattern Detector Module (Conceptual Higher-Order Descriptor Example)
// Enhanced Features:
// - Stores a deeper history of ALU flags using shift registers.
// - Detects MULTIPLE specific "anomalous" patterns across history.
// - Outputs a single "anomaly_detected" flag if ANY pattern matches.
// ======================================================================
module pattern_detector(
input clk,
input reset,
// Current flags represent the flags from the *current* cycle's ALU output (EX stage)
input wire zero_flag_current,
input wire negative_flag_current,
input wire carry_flag_current,
input wire overflow_flag_current,


output wire anomaly_detected_out // Corrected: Changed from 'reg' to 'wire' as it's driven by assign


);


// History depth: We'll store current and previous 2 cycles for 3-cycle total view
parameter HISTORY_DEPTH = 3; // For 3 cycles of data (current, prev1, prev2).

// Shift registers for ALU flags (these must be 'reg' as they are assigned in always block)
reg [HISTORY_DEPTH-1:0] zero_flag_history;
reg [HISTORY_DEPTH-1:0] negative_flag_history;
reg [HISTORY_DEPTH-1:0] carry_flag_history;
reg [HISTORY_DEPTH-1:0] overflow_flag_history;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        zero_flag_history <= 'b0;
        negative_flag_history <= 'b0;
        carry_flag_history <= 'b0;
        overflow_flag_history <= 'b0;
        // anomaly_detected_out <= 1'b0; // Removed: 'anomaly_detected_out' is now a wire, not reg
    end else begin
        // Shift in current flags, pushing older flags out
        zero_flag_history <= {zero_flag_history[HISTORY_DEPTH-2:0], zero_flag_current};
        negative_flag_history <= {negative_flag_history[HISTORY_DEPTH-2:0], negative_flag_current};
        carry_flag_history <= {carry_flag_history[HISTORY_DEPTH-2:0], carry_flag_current};
        overflow_flag_history <= {overflow_flag_history[HISTORY_DEPTH-2:0], overflow_flag_current};
    end
end

// Define Multiple Anomalous Patterns (using current, prev1, prev2 flags)
// Access: {flag_history[0]} is current, {flag_history[1]} is prev1, {flag_history[2]} is prev2

wire pattern1_match;
wire pattern2_match;

// Pattern 1: (Prev2 Zero=0, Prev1 Negative=1, Current Carry=1)
// A pattern that might indicate a specific arithmetic flow leading to a problem
assign pattern1_match = (!zero_flag_history[2]) && (negative_flag_history[1]) && (carry_flag_history[0]);

// Pattern 2: (Prev2 Carry=1, Prev1 Overflow=0, Current Zero=0)
// A pattern that might indicate an unexpected sequence of flags related to overflow/zero conditions
assign pattern2_match = (carry_flag_history[2]) && (!overflow_flag_history[1]) && (!zero_flag_history[0]);

// If ANY defined pattern matches, assert anomaly_detected
assign anomaly_detected_out = pattern1_match || pattern2_match;



endmodule

// ======================================================================
// FSM Entropy Overlay Module
// Description: This module implements a Finite State Machine (FSM) that
//              acts as an overlay, dynamically adjusting system behavior
//              (e.g., stalling, flushing, or locking) based on a
//              combination of machine learning (ML) predictions,
//              internal entropy scores, hazard flags, and various
//              override signals. It's designed to manage system stability
//              and security in the face of unpredictable or anomalous
//              conditions.
//
// States:
// - STATE_OK: Normal operation.
// - STATE_STALL: Halts execution to prevent potential issues, allowing
//                for resolution or re-evaluation.
// - STATE_FLUSH: Clears pipelines or buffers, typically in response to
//                detected corruption or irrecoverable states.
// - STATE_LOCK: Enters a secure, unchangeable state, usually indicating
//               a critical security breach or system integrity compromise.
//
// Inputs:
// - ML Predicted Action: Direct guidance from an ML model on desired
//                        system state.
// - Internal Entropy Score: A measure of randomness or unpredictability
//                           within the system, indicating potential
//                           anomalies or attacks.
// - Internal Hazard Flag: Indicates an architectural hazard within the
//                         system (e.g., data dependency, control hazard).
// - Analog Overrides: External, high-priority signals for immediate
//                     system state changes (lock or flush).
// - Classified Entropy Level: Pre-classified severity of entropy (low, mid, critical).
// - Quantum Override Signal: A highly critical override, possibly from
//                            a quantum-level monitoring system, forcing a lock.
// - Instruction Type: Categorization of the currently executing instruction,
//                     used for context-aware state transitions.
//
// Outputs:
// - FSM State: The current operational state of the FSM.
// - Entropy Log Out: Logs the entropy score when a state transition occurs.
// - Instruction Type Log Out: Logs the instruction type when a state transition occurs.
// ======================================================================
module fsm_entropy_overlay(
input wire clk,                         // Clock signal
input wire rst_n,                       // Asynchronous active-low reset
input wire [1:0] ml_predicted_action,   // Machine Learning model's predicted action
input wire [7:0] internal_entropy_score,// Current internal entropy score
input wire internal_hazard_flag,        // Flag indicating an internal system hazard
input wire analog_lock_override,        // External override to force LOCK state
input wire analog_flush_override,       // External override to force FLUSH state
input wire [1:0] classified_entropy_level, // Pre-classified entropy level (Low, Mid, Critical)
input wire quantum_override_signal,     // Critical override from quantum monitoring
input wire [2:0] instr_type,            // Type of the current instruction
output reg [1:0] fsm_state,             // Current FSM state
output reg [7:0] entropy_log_out,       // Log of entropy score at state change
output reg [2:0] instr_type_log_out     // Log of instruction type at state change
);

```
// --- FSM State Definitions ---
// These parameters define the possible states of the Finite State Machine.
parameter STATE_OK    = 2'b00; // Normal operational state
parameter STATE_STALL = 2'b01; // System stalled, awaiting resolution or re-evaluation
parameter STATE_FLUSH = 2'b10; // System flushing pipelines/buffers
parameter STATE_LOCK  = 2'b11; // System locked due to critical event

// --- ML Action Code Definitions ---
// These parameters define the actions suggested by the Machine Learning model.
parameter ML_OK    = 2'b00; // ML suggests normal operation
parameter ML_STALL = 2'b01; // ML suggests stalling the system
parameter ML_FLUSH = 2'b10; // ML suggests flushing the system
parameter ML_LOCK  = 2'b11; // ML suggests locking the system

// --- Entropy Classification Levels ---
// These parameters categorize the internal entropy score into predefined levels.
parameter ENTROPY_LOW      = 2'b00; // Low entropy, normal
parameter ENTROPY_MID      = 2'b01; // Medium entropy, potentially concerning
parameter ENTROPY_CRITICAL = 2'b10; // Critical entropy, highly concerning

// --- Instruction Type Definitions ---
// These parameters define different categories of instructions processed by the system.
parameter INSTR_TYPE_ALU    = 3'b000; // Arithmetic Logic Unit operation
parameter INSTR_TYPE_LOAD   = 3'b001; // Memory load operation
parameter INSTR_TYPE_STORE  = 3'b010; // Memory store operation
parameter INSTR_TYPE_BRANCH = 3'b011; // Program control branch instruction
parameter INSTR_TYPE_JUMP   = 3'b100; // Program control jump instruction
parameter INSTR_TYPE_OTHER  = 3'b111; // Other/unclassified instruction type

// --- Thresholds ---
// Defines the threshold for what is considered a high entropy score.
parameter ENTROPY_HIGH_THRESHOLD = 8'd180; // Example threshold for high entropy (decimal 180)

// --- Internal FSM State Registers ---
// Registers to hold the current and next state of the FSM for synchronous updates.
reg [1:0] current_state, next_state;

// ==================================================================
// Synchronous State Register Logic
//
// This always block updates the current FSM state and logs relevant
// information (entropy score, instruction type) on state transitions.
// It is sensitive to the positive edge of the clock and the negative
// edge of the asynchronous reset.
// ==================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Asynchronous reset condition:
        // Reset the FSM to the initial 'OK' state and clear log outputs.
        current_state <= STATE_OK;
        entropy_log_out <= 8'h00;
        instr_type_log_out <= INSTR_TYPE_OTHER; // Reset to a known default
    end else begin
        // Synchronous update:
        // Update the current state to the calculated next state.
        current_state <= next_state;

        // Log continuously to aid in debugging, regardless of state change.
        entropy_log_out <= internal_entropy_score;
        instr_type_log_out <= instr_type;
    end
end

// ==================================================================
// Combinational Next-State Logic
//
// This always block determines the next FSM state based on current inputs.
// It's a combinational block, meaning its outputs react immediately
// to changes in its inputs. Priority is given to override signals.
// ==================================================================
always @(*) begin
    // Default to staying in the current state unless a condition dictates a change.
    next_state = current_state;

    // --- High-Priority Overrides ---
    // Quantum override takes highest priority, forcing a LOCK.
    if (quantum_override_signal) begin
        next_state = STATE_LOCK;
    end
    // Analog lock override takes next priority.
    else if (analog_lock_override) begin
        next_state = STATE_LOCK;
    end
    // Analog flush override takes next priority.
    else if (analog_flush_override) begin
        next_state = STATE_FLUSH;
    end
    // --- Normal State Transition Logic ---
    else begin
        case (current_state)
            STATE_OK: begin
                // In OK state, process ML predictions first.
                case (ml_predicted_action)
                    ML_STALL: next_state = STATE_STALL;
                    ML_FLUSH: next_state = STATE_FLUSH;
                    ML_LOCK:  next_state = STATE_LOCK;
                    default: begin
                        // If ML is OK, check internal entropy and hazards.
                        // Prioritize direct high entropy check if classified as LOW or unclassified
                        if ((classified_entropy_level == ENTROPY_LOW || classified_entropy_level == 2'b11) && // 2'b11 for unclassified
                            internal_entropy_score > ENTROPY_HIGH_THRESHOLD) begin
                            next_state = STATE_STALL; // High entropy -> STALL
                        end else if (internal_hazard_flag) begin
                            next_state = STATE_STALL; // Internal hazard -> STALL
                        end else if (classified_entropy_level == ENTROPY_CRITICAL) begin
                            // Critical entropy without ML intervention -> FLUSH or STALL based on instr_type
                            case (instr_type)
                                INSTR_TYPE_BRANCH, INSTR_TYPE_JUMP: next_state = STATE_STALL;
                                INSTR_TYPE_LOAD, INSTR_TYPE_STORE: next_state = STATE_FLUSH;
                                default: next_state = STATE_FLUSH;
                            endcase
                        end else if (classified_entropy_level == ENTROPY_MID) begin
                            // Medium entropy: more conservative action
                            case (instr_type)
                                INSTR_TYPE_BRANCH, INSTR_TYPE_JUMP, INSTR_TYPE_LOAD, INSTR_TYPE_STORE: next_state = STATE_STALL;
                                default: next_state = STATE_OK; // Stay OK for other types with mid entropy
                            endcase
                        end
                        // Otherwise, stay in OK
                    end
                endcase
            end
            STATE_STALL: begin
                // In STALL state, ML can transition to FLUSH or LOCK, or return to OK if conditions clear.
                case (ml_predicted_action)
                    ML_FLUSH: next_state = STATE_FLUSH;
                    ML_LOCK:  next_state = STATE_LOCK;
                    default: begin
                        // Return to OK if ML is OK, no hazard, and entropy is low.
                        if (ml_predicted_action == ML_OK &&
                            !internal_hazard_flag &&
                            classified_entropy_level == ENTROPY_LOW &&
                            internal_entropy_score <= ENTROPY_HIGH_THRESHOLD) // Use <= for clear exit
                            next_state = STATE_OK;
                        end
                endcase
            end
            STATE_FLUSH: begin
                // In FLUSH state, ML can transition to LOCK, or return to OK/STALL if conditions clear.
                case (ml_predicted_action)
                    ML_LOCK: next_state = STATE_LOCK;
                    default: begin
                        // Return to OK if ML is OK, no hazard, and entropy is low.
                        if (ml_predicted_action == ML_OK &&
                            !internal_hazard_flag &&
                            classified_entropy_level == ENTROPY_LOW &&
                            internal_entropy_score <= ENTROPY_HIGH_THRESHOLD) // Use <= for clear exit
                            next_state = STATE_OK;
                        // If ML suggests STALL, transition to STALL.
                        else if (ml_predicted_action == ML_STALL)
                            next_state = STATE_STALL;
                    end
                endcase
            end
            STATE_LOCK: begin
                // NEW: Allow exit from LOCK state if all lock-forcing overrides are removed
                // AND no critical entropy or internal hazard is present.
                if (!quantum_override_signal && !analog_lock_override &&
                    classified_entropy_level != ENTROPY_CRITICAL && !internal_hazard_flag) begin
                    next_state = STATE_OK; // Return to OK if conditions clear
                end else begin
                    next_state = STATE_LOCK; // Remain locked otherwise
                end
            end
            default:
                // Safety default: if in an undefined state, return to OK.
                next_state = STATE_OK;
        endcase
    end
end

// ==================================================================
// Output State Assignment
//
// This always block continuously assigns the current FSM state to the
// output register `fsm_state`. This ensures that the external world
// always sees the current operational state of the module.
// ==================================================================
always @(*) begin
    fsm_state = current_state;
end



endmodule

// ===============================================================================
// ARCHON HAZARD OVERRIDE UNIT (AHO) - Integrated and Enhanced
// Purpose: This module implements the Archon Hazard Override (AHO) unit,
//          responsible for detecting hazardous internal states and generating
//          override signals (flush, stall) for the CPU pipeline.
//
// Key Enhancements:
// 1. Direct incorporation of 'cache_miss_rate_tracker' as a primary input.
// 2. Implementation of 'fluctuating impact' for various metrics through
//    dynamic weighting, controlled by an external 'ml_predicted_action'.
// 3. A sophisticated rule-based decision engine for hazard mitigation,
//    combining dynamically weighted scores with fixed-priority anomaly detection.
// This version is designed to provide 'override_flush_sig' and 'override_stall_sig'
// to the Probabilistic Hazard FSM, rather than direct pipeline control.
// ===============================================================================

module archon_hazard_override_unit (
input wire                 clk, // Changed from logic to wire
input wire                 rst_n, // Active low reset // Changed from logic to wire


// Core Hazard Metrics (now adapted to 8-bit where needed, Chaos is 16-bit)
input wire [7:0]           internal_entropy_score_val, // Changed from logic to wire
input wire [15:0]          chaos_score_val,            // Changed from logic to wire
input wire                 anomaly_detected_val,       // Changed from logic to wire

// Performance/System Health Metrics (now adapted to 8-bit where needed)
input wire [7:0]           branch_miss_rate_tracker,   // Changed from logic to wire
input wire [7:0]           cache_miss_rate_tracker,    // NEW: Current cache miss rate (from Data Memory/Cache) // Changed from logic to wire
input wire [7:0]           exec_pressure_tracker,      // Current execution pressure (e.g., pipeline fullness) // Changed from logic to wire

// Input from external ML model for dynamic weighting/context
// This input dictates the current 'risk posture' or 'mode' for hazard detection.
// Examples: 2'b00=Normal, 2'b01=MonitorRisk, 2'b10=HighRisk, 2'b11=CriticalRisk
input wire [1:0]           ml_predicted_action, // Changed from logic to wire

// Dynamically scaled thresholds for the combined hazard score (adjusted for new total score range)
// These thresholds would typically be provided by an external control unit or derived
// from system-wide context/ML predictions, scaled appropriately for 'total_combined_hazard_score'.
input wire [20:0]          scaled_flush_threshold,     // Changed from logic to wire
input wire [20:0]          scaled_stall_threshold,     // Changed from logic to wire

// Outputs to CPU pipeline control (specifically for Probabilistic Hazard FSM or main control)
output reg                override_flush_sig,         // Request for CPU pipeline flush // Changed from logic to reg
output reg                override_stall_sig,         // Request for CPU pipeline stall // Changed from logic to reg
output reg [1:0]          hazard_detected_level       // Severity: 00=None, 01=Low, 10=Medium, 11=High/Critical // Changed from logic to reg


);


// --- Internal Signals for Dynamic Weight Assignment (Fluctuating Impact) ---
// These 4-bit weights (0-15) are dynamically adjusted based on 'ml_predicted_action'.
// They amplify or de-emphasize the impact of each raw metric on the total hazard score.
reg [3:0] W_entropy; // Changed from logic to reg
reg [3:0] W_chaos; // Changed from logic to reg
reg [3:0] W_branch; // Changed from logic to reg
reg [3:0] W_cache; // Changed from logic to reg
reg [3:0] W_exec; // Changed from logic to reg

// --- Internal Signals for Weighted Scores ---
// Individual weighted scores are calculated by multiplying raw scores by weights.
// Max product for 8-bit * 4-bit: 255 * 15 = 3825. A 12-bit register is sufficient.
// Max product for 16-bit * 4-bit: 65535 * 15 = 983025. A 20-bit register is sufficient.
wire [11:0] weighted_entropy_score;   // Changed from logic to wire
wire [19:0] weighted_chaos_score;     // Changed from logic to wire
wire [11:0] weighted_branch_miss_score; // Changed from logic to wire
wire [11:0] weighted_cache_miss_score;  // Changed from logic to wire
wire [11:0] weighted_exec_pressure_score; // Changed from logic to wire

// --- Total Combined Hazard Score ---
// Sum of all weighted scores.
// Max sum: (3 * 3825) + (2 * 983025) = 11475 + 1966050 = 1977525.
// A 21-bit register is sufficient (max value 2097151).
wire [20:0] total_combined_hazard_score; // Changed from logic to wire

// --- Output Registers (for synchronous outputs) ---
// These are now directly the output ports, declared as `reg`
// No separate `reg reg_override_flush_sig;` etc. needed.

// --- Clocked Logic for Output Registers ---
// This block is only needed if the outputs are synchronous registers
// Since override_flush_sig and override_stall_sig are combinational outputs
// from the always @(*) block, they should be assigned directly there.
// The hazard_detected_level is also combinational from that same block.
// The previous 'reg reg_override_flush_sig;' and associated always block
// were for registering combinational signals at module output.
// Since the outputs themselves are now `reg`, this block is removed
// and the assignments are moved into the always @(*) block directly.

// --- Combinational Logic for Dynamic Weight Assignment (Fluctuating Impact) ---
// This block determines the importance (weights) of each metric based on the
// 'ml_predicted_action', allowing the system to adapt its sensitivity.
always @(*) begin
    case (ml_predicted_action)
        2'b00: begin // Normal Operation: Balanced weights, general monitoring
            W_entropy = 4'd8;   // Moderate impact for entropy/chaos
            W_chaos   = 4'd7;
            W_branch  = 4'd5;   // Moderate for branch/cache misses (performance indicators)
            W_cache   = 4'd6;
            W_exec    = 4'd4;   // Lower for execution pressure
        end
        2'b01: begin // Monitor Risk: Increased focus on anomaly/chaos indicators
            W_entropy = 4'd10; // Higher impact for entropy/chaos
            W_chaos   = 4'd9;
            W_branch  = 4'd7;   // Slightly increased for branch/cache misses
            W_cache   = 4'd8;
            W_exec    = 4'd3;   // Reduced emphasis on exec pressure
        end
        2'b10: begin // High Risk: Strong emphasis on potential security/stability issues
            W_entropy = 4'd12; // Significantly higher impact for entropy/chaos
            W_chaos   = 4'd11;
            W_branch  = 4'd9;   // Substantially increased for branch/cache misses (could indicate attack)
            W_cache   = 4'd10;
            W_exec    = 4'd2;   // Minimal emphasis on general performance for immediate risk
        end
        2'b11: begin // Critical Risk: Maximum sensitivity for all hazard indicators
            W_entropy = 4'd15; // Max impact
            W_chaos   = 4'd15; // Max impact
            W_branch  = 4'd13; // Very high impact
            W_cache    = 4'd14; // Very high impact
            W_exec    = 4'd1;   // Almost no impact for exec pressure, focus is on stopping threat
        end
        default: begin // Defensive default: Fallback to normal operation weights
            W_entropy = 4'd8; W_chaos = 4'd7; W_branch = 4'd5; W_cache = 4'd6; W_exec = 4'd4;
        end
    endcase
end

// --- Combinational Logic for Weighted Score Calculation (Dynamic Weighted Sum) ---
// Each raw score is multiplied by its dynamically determined weight.
assign weighted_entropy_score   = internal_entropy_score_val * W_entropy;
assign weighted_chaos_score     = chaos_score_val * W_chaos;
assign weighted_branch_miss_score = branch_miss_rate_tracker * W_branch;
assign weighted_cache_miss_score    = cache_miss_rate_tracker * W_cache; // NEW: Cache miss included
assign weighted_exec_pressure_score = exec_pressure_tracker * W_exec;

// The total combined hazard score aggregates all weighted metric impacts.
assign total_combined_hazard_score =
    weighted_entropy_score +
    weighted_chaos_score +
    weighted_branch_miss_score +
    weighted_cache_miss_score +
    weighted_exec_pressure_score;

// --- Combinational Logic for Override Signals (Multi-dimensional Rule Engine) ---
// This block implements the decision logic, prioritizing different hazard indicators.
always @(*) begin
    override_flush_sig = 1'b0; // Assign directly to output reg
    override_stall_sig = 1'b0; // Assign directly to output reg
    hazard_detected_level = 2'b00; // Default to no hazard

    // Rule 1: High-priority anomaly detection (Pattern Detector)
    // If an anomaly is detected, this should trigger a flush immediately,
    // regardless of the combined hazard score, as it signifies a critical state.
    if (anomaly_detected_val) begin
        override_flush_sig = 1'b1;
        hazard_detected_level = 2'b11; // Critical
    end else begin
        // Rule 2: Evaluate based on combined hazard score against dynamic thresholds
        if (total_combined_hazard_score > scaled_flush_threshold) begin
            override_flush_sig = 1'b1;
            hazard_detected_level = 2'b10; // Medium to High (depending on threshold severity)
        end else if (total_combined_hazard_score > scaled_stall_threshold) begin
            override_stall_sig = 1'b1;
            hazard_detected_level = 2'b01; // Low to Medium
        end else begin
            // No significant hazard detected by AHO's scoring system
            override_flush_sig = 1'b0;
            override_stall_sig = 1'b0;
            hazard_detected_level = 2'b00; // None
        end
    end
end


endmodule

// ===============================================================================
// NEW MODULE: entropy_control_logic.v
// Purpose: Directly uses the 16-bit external entropy input from 'entropy_bus.txt'.
//          Applies simple, configurable thresholds to generate stall/flush signals.
//          This module provides the *base* entropy-driven control signals.
//          These can then be modulated by ML and chaos predictors in the main CPU.
// ===============================================================================
module entropy_control_logic(
input wire [15:0] external_entropy_in, // 16-bit external entropy from entropy_bus.txt
output wire entropy_stall,            // Assert to signal a basic entropy-induced stall
output wire entropy_flush             // Assert to signal a basic entropy-induced flush
);


// Define entropy thresholds for stall and flush
// These values are for a 16-bit (0-65535) entropy input.
// Changed thresholds to be more sensitive to higher values, and logic to trigger on > threshold.
parameter ENTROPY_STALL_THRESHOLD = 16'd10000;  // Example: Above 10000, consider stalling
parameter ENTROPY_FLUSH_THRESHOLD = 16'd50000; // Example: Above 50000, consider flushing

// Inverted logic: now asserts stall/flush when entropy_in is GREATER than threshold
assign entropy_stall = (external_entropy_in > ENTROPY_STALL_THRESHOLD);
assign entropy_flush = (external_entropy_in > ENTROPY_FLUSH_THRESHOLD);


endmodule

// ===============================================================================
// NEW MODULE: entropy_trigger_decoder.v
// Purpose: Simulates compression of incoming analog entropy signals (8-bit)
//          into meaningful trigger vectors or score levels (2-bit).
// ===============================================================================
module entropy_trigger_decoder(
input wire [7:0] entropy_in,    // 8-bit entropy score (0-255)
output reg [1:0] signal_class   // 2-bit output: 00 = LOW, 01 = MID, 10 = CRITICAL
);


// Define thresholds for classification
parameter THRESHOLD_LOW_TO_MID = 8'd85;     // Up to 85 is LOW
parameter THRESHOLD_MID_TO_CRITICAL = 8'd170; // Up to 170 is MID, above is CRITICAL

always @(*) begin
    if (entropy_in <= THRESHOLD_LOW_TO_MID) begin
        signal_class = 2'b00; // LOW
    end else if (entropy_in <= THRESHOLD_MID_TO_CRITICAL) begin
        signal_class = 2'b01; // MID
    end else begin
        signal_class = 2'b10; // CRITICAL
    end
end



endmodule

module pipeline_cpu(
input wire clk,
input wire reset, // Active high reset (converts to active low for some modules)
input wire [15:0] external_entropy_in, // Input from entropy_bus.txt (for Entropy Control Logic)
input wire [1:0] ml_predicted_action, // ML model's predicted action for AHO and FSM
input wire internal_hazard_flag_for_fsm, // This is an input to pipeline_cpu from archon_top


// START OF ADDED PARTS: Analog Override Inputs for pipeline_cpu and Quantum Override
input wire analog_lock_override_in,  // From top-level analog controller
input wire analog_flush_override_in, // From top-level analog controller
input wire quantum_override_signal_in, // NEW: Quantum override signal from Qiskit simulation
// END OF ADDED PARTS

output wire [3:0] debug_pc,         // For debugging: current PC
output wire [15:0] debug_instr,     // For debugging: current instruction
output wire debug_stall,            // For debugging: indicates pipeline stall
output wire debug_flush,            // For debugging: indicates pipeline flush
output wire debug_lock,             // For debugging: indicates system lock
output wire [7:0] debug_fsm_entropy_log, // For debugging: entropy value logged by new FSM
output wire [2:0] debug_fsm_instr_type_log, // NEW: Debug output for logged instruction type
output wire debug_hazard_flag,      // NEW OUTPUT: Expose the internal hazard flag
output wire [1:0] debug_fsm_state   // NEW OUTPUT: Expose the FSM state



);


// --- Active Low Reset for Modules that use it ---
wire rst_n = ~reset;

// --- Internal Wires & Registers for Pipeline Stages ---
reg [3:0] pc_reg;
wire [15:0] if_instr;
wire [3:0] if_pc_plus_1;
reg [3:0] next_pc;

reg [3:0] if_id_pc_plus_1_reg;
reg [15:0] if_id_instr_reg;

wire [3:0] id_pc_plus_1;
wire [15:0] id_instr;
wire [3:0] id_operand1;
wire [3:0] id_operand2;
wire [2:0] id_rs1_addr;
wire [2:0] id_rs2_addr;
wire [2:0] id_rd_addr;
reg [2:0] id_alu_op;
wire [3:0] id_immediate;
wire id_reg_write_enable;
wire id_mem_read_enable;
wire id_mem_write_enable;
wire id_is_branch_inst;
wire id_is_jump_inst;
wire [3:0] id_branch_target;

// CHANGED: From wire to reg for combinational assignment in always @(*) block
reg [2:0] instr_type_to_fsm_comb;
reg [2:0] instr_type_to_fsm_reg;   // Registered instruction type for FSM input

wire [3:0] ex_opcode_from_ex_stage; // Wire to hold opcode from EX stage

reg [3:0] id_ex_pc_plus_1_reg;
reg [3:0] id_ex_operand1_reg;
reg [3:0] id_ex_operand2_reg;
reg [2:0] id_ex_rd_addr_reg;
reg [2:0] id_ex_alu_op_reg;
reg id_ex_reg_write_enable_reg;
reg id_ex_mem_read_enable_reg;
reg id_ex_mem_write_enable_reg;
reg id_ex_is_branch_inst_reg;
reg id_ex_is_jump_inst_reg;
reg [3:0] id_ex_branch_target_reg;
reg [15:0] id_ex_instr_reg;

wire [3:0] ex_alu_operand1;
wire [3:0] ex_alu_operand2;
wire [3:0] ex_alu_result;
wire ex_zero_flag;
wire ex_negative_flag;
wire ex_carry_flag;
wire ex_overflow_flag;
wire [2:0] ex_rd_addr;
wire ex_reg_write_enable;
wire ex_mem_read_enable;
wire ex_mem_write_enable;
wire ex_is_branch_inst;
wire ex_is_jump_inst;
wire [3:0] ex_branch_target;
wire [3:0] ex_branch_pc;

reg [3:0] ex_mem_alu_result_reg;
reg [3:0] ex_mem_mem_write_data_reg;
reg [2:0] ex_mem_rd_addr_reg;
reg ex_mem_reg_write_enable_reg;
reg ex_mem_mem_read_enable_reg;
reg ex_mem_mem_write_enable_reg;
reg ex_mem_zero_flag_reg;
reg ex_mem_is_branch_inst_reg;
reg ex_mem_is_jump_inst_reg;
reg [3:0] ex_mem_pc_plus_1_reg;
reg [3:0] ex_mem_branch_target_reg;
reg [3:0] ex_mem_branch_pc_reg;

wire [3:0] mem_read_data;
wire [3:0] mem_alu_result;
wire [2:0] mem_rd_addr;
wire mem_reg_write_enable;
wire mem_mem_read_enable;
wire mem_mem_write_enable;
wire [3:0] mem_mem_addr;

wire branch_actual_taken;
wire branch_mispredicted_local;
reg branch_mispredicted;
wire [3:0] branch_resolved_pc;
wire [3:0] branch_resolved_target_pc;

reg [3:0] mem_wb_write_data_reg;
reg [2:0] mem_wb_rd_addr_reg;
reg mem_wb_reg_write_enable_reg;

wire [3:0] wb_write_data;
wire [2:0] wb_rd_addr;
wire wb_reg_write_enable;

// --- Pipeline Control Signals ---
wire pipeline_stall;
wire pipeline_flush;

// For simplicity, tracking rough execution pressure
reg [7:0] exec_pressure_counter;
reg [7:0] cache_miss_rate_dummy;

// AHO internal hazard signals
wire aho_override_flush_req;
wire aho_override_stall_req;
wire [1:0] aho_hazard_level;

// Consolidated internal hazard flag for the new FSM
wire new_fsm_internal_hazard_flag;
wire [1:0] new_fsm_control_signal;
wire [7:0] new_fsm_entropy_log;
wire [2:0] new_fsm_instr_type_log;

localparam AHO_SCALED_FLUSH_THRESH = 21'd1000000;
localparam AHO_SCALED_STALL_THRESH = 21'd500000;

wire [1:0] classified_entropy_level_wire;

// --- Explicit declaration for debug_branch_miss_rate ---
wire [7:0] debug_branch_miss_rate;
reg [7:0] branch_miss_rate_counter; // Declared once here

// --- Instantiate Sub-modules ---
instruction_ram i_imem (
    .clk(clk),
    .reset(reset),
    .pc_in(pc_reg),
    .instr_opcode(if_instr)
);

register_file i_regfile (
    .clk(clk),
    .reset(reset),
    .regfile_write_enable(wb_reg_write_enable),
    .write_addr(wb_rd_addr),
    .write_data(wb_write_data),
    .read_addr1(id_rs1_addr),
    .read_addr2(id_rs2_addr),
    .read_data1(id_operand1),
    .read_data2(id_operand2)
);

alu_unit i_alu (
    .alu_operand1(ex_alu_operand1),
    .alu_operand2(ex_alu_operand2),
    .alu_op(id_ex_alu_op_reg),
    .alu_result(ex_alu_result),
    .zero_flag(ex_zero_flag),
    .negative_flag(ex_negative_flag),
    .carry_flag(ex_carry_flag),
    .overflow_flag(ex_overflow_flag)
);

data_mem i_dmem (
    .clk(clk),
    .mem_write_enable(mem_mem_write_enable),
    .mem_read_enable(mem_mem_read_enable),
    .addr(mem_mem_addr),
    .write_data(ex_mem_mem_write_data_reg),
    .read_data(mem_read_data)
);

wire [3:0] if_btb_predicted_next_pc;
wire if_btb_predicted_taken;
branch_target_buffer i_btb (
    .clk(clk),
    .reset(reset),
    .pc_in(pc_reg),
    .branch_resolved_pc(branch_resolved_pc),
    .branch_resolved_pc_valid(ex_mem_is_branch_inst_reg || ex_mem_is_jump_inst_reg),
    .branch_resolved_target_pc(branch_resolved_target_pc),
    .branch_resolved_taken(branch_actual_taken),
    .predicted_next_pc(if_btb_predicted_next_pc),
    .predicted_taken(if_btb_predicted_taken)
);

wire [3:0] qed_instr_opcode_input;
assign qed_instr_opcode_input = id_ex_instr_reg[15:12];
wire qed_reset = reset;
wire [7:0] qed_entropy_score_out;
quantum_entropy_detector i_qed (
    .clk(clk),
    .reset(qed_reset),
    .instr_opcode(qed_instr_opcode_input),
    .alu_result(ex_alu_result),
    .zero_flag(ex_zero_flag),
    .entropy_score_out(qed_entropy_score_out)
);

wire cd_reset = reset;
wire [15:0] cd_chaos_score_out;
chaos_detector i_chaos_detector (
    .clk(clk),
    .reset(cd_reset),
    .branch_mispredicted(branch_mispredicted),
    .mem_access_addr(mem_mem_addr),
    .data_mem_read_data(mem_read_data),
    .chaos_score_out(cd_chaos_score_out)
);

wire pd_reset = reset;
wire pd_anomaly_detected_out;
pattern_detector i_pattern_detector (
    .clk(clk),
    .reset(pd_reset),
    .zero_flag_current(ex_zero_flag),
    .negative_flag_current(ex_negative_flag),
    .carry_flag_current(ex_carry_flag),
    .overflow_flag_current(ex_overflow_flag),
    .anomaly_detected_out(pd_anomaly_detected_out)
);

// This block is the sole driver for branch_miss_rate_counter
always @(posedge clk or posedge reset) begin
    if (reset) begin
        branch_miss_rate_counter <= 8'h0;
    end else if (branch_mispredicted) begin
        if (branch_miss_rate_counter < 8'hFF) begin
            branch_miss_rate_counter <= branch_miss_rate_counter + 8'h1;
        end
    end else begin
        if (branch_miss_rate_counter > 8'h0) begin
            branch_miss_rate_counter <= branch_miss_rate_counter - 8'h01; // Decay
        end
    end
end

assign debug_branch_miss_rate = branch_miss_rate_counter;

archon_hazard_override_unit i_aho (
    .clk(clk),
    .rst_n(rst_n),
    .internal_entropy_score_val(qed_entropy_score_out),
    .chaos_score_val(cd_chaos_score_out),
    .anomaly_detected_val(pd_anomaly_detected_out),
    .branch_miss_rate_tracker(debug_branch_miss_rate),
    .cache_miss_rate_tracker(cache_miss_rate_dummy),
    .exec_pressure_tracker(exec_pressure_counter),
    .ml_predicted_action(ml_predicted_action),
    .scaled_flush_threshold(AHO_SCALED_FLUSH_THRESH),
    .scaled_stall_threshold(AHO_SCALED_STALL_THRESH),
    .override_flush_sig(aho_override_flush_req),
    .override_stall_sig(aho_override_stall_req),
    .hazard_detected_level(aho_hazard_level)
);

entropy_trigger_decoder i_entropy_decoder (
    .entropy_in(qed_entropy_score_out),
    .signal_class(classified_entropy_level_wire)
);

// Assign the opcode from the EX stage for FSM instruction type logging
assign ex_opcode_from_ex_stage = id_ex_instr_reg[15:12];

// Combinational assignment for instruction type
always @(*) begin
    case (ex_opcode_from_ex_stage)
        4'h1, 4'h2, 4'h3, 4'h6: instr_type_to_fsm_comb = 3'b000; // ALU
        4'h4:                   instr_type_to_fsm_comb = 3'b001; // LOAD
        4'h5:                   instr_type_to_fsm_comb = 3'b010; // STORE
        4'h7:                   instr_type_to_fsm_comb = 3'b011; // BRANCH
        4'h8:                   instr_type_to_fsm_comb = 3'b100; // JUMP
        default:                instr_type_to_fsm_comb = 3'b111; // OTHER (e.g., NOP or unmapped)
    endcase
end

// Synchronously register the instruction type for FSM input
always @(posedge clk or posedge reset) begin
    if (reset) begin
        instr_type_to_fsm_reg <= 3'b111; // Default to OTHER on reset
    end else begin
        instr_type_to_fsm_reg <= instr_type_to_fsm_comb;
    end
end

// NEW: Entropy-Aware FSM
// Consolidate AHO's requests with the 'internal_hazard_flag_for_fsm' input
assign new_fsm_internal_hazard_flag = aho_override_flush_req || aho_override_stall_req || internal_hazard_flag_for_fsm;

fsm_entropy_overlay i_entropy_fsm (
    .clk(clk),
    .rst_n(rst_n),
    .ml_predicted_action(ml_predicted_action),
    .internal_entropy_score(qed_entropy_score_out),
    .internal_hazard_flag(new_fsm_internal_hazard_flag),
    .analog_lock_override(analog_lock_override_in),
    .analog_flush_override(analog_flush_override_in),
    .classified_entropy_level(classified_entropy_level_wire),
    .quantum_override_signal(quantum_override_signal_in),
    .instr_type(instr_type_to_fsm_reg), // CHANGED: Use the registered version
    .fsm_state(new_fsm_control_signal),
    .entropy_log_out(new_fsm_entropy_log),
    .instr_type_log_out(new_fsm_instr_type_log)
);

wire entropy_ctrl_stall;
wire entropy_ctrl_flush;
entropy_control_logic i_entropy_ctrl (
    .external_entropy_in(external_entropy_in),
    .entropy_stall(entropy_ctrl_stall),
    .entropy_flush(entropy_ctrl_flush)
);

// --- Pipeline Control Unit ---
assign pipeline_flush = (new_fsm_control_signal == 2'b10) ||
                        (new_fsm_control_signal == 2'b11) ||
                        entropy_ctrl_flush;

assign pipeline_stall = (new_fsm_control_signal == 2'b01) ||
                        (new_fsm_control_signal == 2'b11) ||
                        entropy_ctrl_stall;

// --- Execution Pressure Counter ---
always @(posedge clk or posedge reset) begin
    if (reset) begin
        exec_pressure_counter <= 8'h0;
        cache_miss_rate_dummy <= 8'h0;
    end else if (pipeline_flush) begin
        exec_pressure_counter <= 8'h0;
        cache_miss_rate_dummy <= 8'h0;
    end else if (pipeline_stall) begin
        exec_pressure_counter <= exec_pressure_counter;
        cache_miss_rate_dummy <= cache_miss_rate_dummy;
    end else begin
        if (if_id_instr_reg[15:12] != 4'h9) begin
            if (exec_pressure_counter < 8'hFF) begin
                exec_pressure_counter <= exec_pressure_counter + 8'h1;
            end
        end else begin
            if (exec_pressure_counter > 8'h0) begin
                exec_pressure_counter <= exec_pressure_counter - 8'h1;
            end
        end
        // Replaced $urandom_range with a simple, deterministic counter for simulation.
        // This will make cache_miss_rate_dummy increment/decrement predictably.
        // The value will increment by 1 every 5 execution pressure units, and decrement by 1 every 7 units.
        if (cache_miss_rate_dummy < 8'hFF && (exec_pressure_counter % 5 == 0)) begin
            cache_miss_rate_dummy <= cache_miss_rate_dummy + 1;
        end else if (cache_miss_rate_dummy > 8'h0 && (exec_pressure_counter % 7 == 0)) begin
            cache_miss_rate_dummy <= cache_miss_rate_dummy - 1;
        end
    end
end

// --- IF Stage ---
assign if_pc_plus_1 = pc_reg + 4'b0001;
always @(posedge clk or posedge reset) begin
    if (reset) begin
        pc_reg <= 4'h0;
    end else begin
        pc_reg <= next_pc;
    end
end
// Corrected next_pc logic for proper prioritization of control signals
always @(*) begin
    next_pc = if_pc_plus_1; // Default to incrementing PC

    // Prioritize pipeline control signals: Flush then Stall
    if (pipeline_flush) begin
        next_pc = 4'h0; // Reset PC to 0 on flush
    end else if (pipeline_stall) begin
        next_pc = pc_reg; // Hold PC on stall
    end else begin
        // If not stalled or flushed, apply branch/jump logic
        if (ex_mem_is_jump_inst_reg) begin
            next_pc = ex_mem_branch_target_reg;
        end else if (ex_mem_is_branch_inst_reg) begin
            if (branch_actual_taken) begin
                next_pc = ex_mem_branch_target_reg;
            end else begin
                next_pc = ex_mem_pc_plus_1_reg;
            end
        end else if (if_btb_predicted_taken) begin
            next_pc = if_btb_predicted_next_pc;
        end
        // Removed redundant FSM state checks here, as they are covered by pipeline_flush/stall
    end
end
always @(posedge clk or posedge reset) begin
    if (reset || pipeline_flush) begin
        if_id_pc_plus_1_reg <= 4'h0;
        if_id_instr_reg <= 16'h0000;
    end else if (~pipeline_stall) begin
        if_id_pc_plus_1_reg <= if_pc_plus_1;
        if_id_instr_reg <= if_instr;
    end
end

// --- ID Stage ---
assign id_pc_plus_1 = if_id_pc_plus_1_reg;
assign id_instr = if_id_instr_reg;

wire [3:0] id_opcode = id_instr[15:12];
assign id_rd_addr = id_instr[11:9];
assign id_rs1_addr = id_instr[8:6];
assign id_rs2_addr = id_instr[5:3];
assign id_immediate = {1'b0, id_instr[2:0]};

assign id_reg_write_enable = (id_opcode == 4'h1 || id_opcode == 4'h2 || id_opcode == 4'h3 ||
                              id_opcode == 4'h4 || id_opcode == 4'h6 || id_opcode == 4'h0);
assign id_mem_read_enable  = (id_opcode == 4'h4);
assign id_mem_write_enable = (id_opcode == 4'h5);
assign id_is_branch_inst   = (id_opcode == 4'h7);
assign id_is_jump_inst     = (id_opcode == 4'h8);
assign id_branch_target    = id_instr[3:0];

always @(*) begin
    case (id_opcode)
        4'h1: id_alu_op = 3'b000;
        4'h2: id_alu_op = 3'b000;
        4'h3: id_alu_op = 3'b001;
        4'h4: id_alu_op = 3'b000;
        4'h5: id_alu_op = 3'b000;
        4'h6: id_alu_op = 3'b100;
        4'h7: id_alu_op = 3'b001;
        default: id_alu_op = 3'b000;
    endcase
end

wire ex_mem_writes_to_rs1_id = ex_mem_reg_write_enable_reg && (ex_mem_rd_addr_reg == id_rs1_addr);
wire ex_mem_writes_to_rs2_id = ex_mem_reg_write_enable_reg && (ex_mem_rd_addr_reg == id_rs2_addr);
wire mem_wb_writes_to_rs1_id = mem_wb_reg_write_enable_reg && (mem_wb_rd_addr_reg == id_rs1_addr);
wire mem_wb_writes_to_rs2_id = mem_wb_reg_write_enable_reg && (mem_wb_rd_addr_reg == id_rs2_addr);

wire [3:0] forward_operand1 = (ex_mem_writes_to_rs1_id && (id_rs1_addr != 3'b000)) ? ex_mem_alu_result_reg :
                              (mem_wb_writes_to_rs1_id && (id_rs1_addr != 3'b000)) ? mem_wb_write_data_reg :
                              id_operand1;
wire [3:0] forward_operand2 = (ex_mem_writes_to_rs2_id && (id_rs2_addr != 3'b000)) ? ex_mem_alu_result_reg :
                              (mem_wb_writes_to_rs2_id && (id_rs2_addr != 3'b000)) ? mem_wb_write_data_reg :
                              id_operand2;

always @(posedge clk or posedge reset) begin
    if (reset || pipeline_flush) begin
        id_ex_pc_plus_1_reg <= 4'h0;
        id_ex_operand1_reg <= 4'h0;
        id_ex_operand2_reg <= 4'h0;
        id_ex_rd_addr_reg <= 3'h0;
        id_ex_alu_op_reg <= 3'h0;
        id_ex_reg_write_enable_reg <= 1'b0;
        id_ex_mem_read_enable_reg <= 1'b0;
        id_ex_mem_write_enable_reg <= 1'b0;
        id_ex_is_branch_inst_reg <= 1'b0;
        id_ex_is_jump_inst_reg <= 1'b0;
        id_ex_branch_target_reg <= 4'h0;
        id_ex_instr_reg <= 16'h0000;
    end else if (~pipeline_stall) begin
        id_ex_pc_plus_1_reg <= id_pc_plus_1;
        id_ex_operand1_reg <= forward_operand1;
        id_ex_operand2_reg <= (id_opcode == 4'h2 || id_opcode == 4'h4 || id_opcode == 4'h5) ? id_immediate : forward_operand2;
        id_ex_rd_addr_reg <= id_rd_addr;
        id_ex_alu_op_reg <= id_alu_op;
        id_ex_reg_write_enable_reg <= id_reg_write_enable;
        id_ex_mem_read_enable_reg <= id_mem_read_enable;
        id_ex_mem_write_enable_reg <= id_mem_write_enable;
        id_ex_is_branch_inst_reg <= id_is_branch_inst;
        id_ex_is_jump_inst_reg <= id_is_jump_inst;
        id_ex_branch_target_reg <= id_branch_target;
        id_ex_instr_reg <= id_instr;
    end
end

// --- EX Stage ---
assign ex_alu_operand1 = id_ex_operand1_reg;
assign ex_alu_operand2 = id_ex_operand2_reg;
assign ex_rd_addr = id_ex_rd_addr_reg;
assign ex_reg_write_enable = id_ex_reg_write_enable_reg;
assign ex_mem_read_enable = id_ex_mem_read_enable_reg;
assign ex_mem_write_enable = id_ex_mem_write_enable_reg;
assign ex_is_branch_inst = id_ex_is_branch_inst_reg;
assign ex_is_jump_inst = id_ex_is_jump_inst_reg;
assign ex_branch_target = id_ex_branch_target_reg;
assign ex_branch_pc = id_ex_pc_plus_1_reg - 4'b0001;

wire [3:0] actual_branch_target_calc;
assign actual_branch_target_calc = ex_branch_pc + ex_branch_target;

always @(posedge clk or posedge reset) begin
    if (reset || pipeline_flush) begin
        ex_mem_alu_result_reg <= 4'h0;
        ex_mem_mem_write_data_reg <= 4'h0;
        ex_mem_rd_addr_reg <= 3'h0;
        ex_mem_reg_write_enable_reg <= 1'b0;
        ex_mem_mem_read_enable_reg <= 1'b0;
        ex_mem_mem_write_enable_reg <= 1'b0;
        ex_mem_zero_flag_reg <= 1'b0;
        ex_mem_is_branch_inst_reg <= 1'b0;
        ex_mem_is_jump_inst_reg <= 1'b0;
        ex_mem_pc_plus_1_reg <= 4'h0;
        ex_mem_branch_target_reg <= 4'h0;
        ex_mem_branch_pc_reg <= 4'h0;
    end else if (~pipeline_stall) begin
        ex_mem_alu_result_reg <= ex_alu_result;
        ex_mem_mem_write_data_reg <= ex_alu_operand2;
        ex_mem_rd_addr_reg <= ex_rd_addr;
        ex_mem_reg_write_enable_reg <= ex_reg_write_enable;
        ex_mem_mem_read_enable_reg <= ex_mem_read_enable;
        ex_mem_mem_write_enable_reg <= ex_mem_write_enable;
        ex_mem_zero_flag_reg <= ex_zero_flag;
        ex_mem_is_branch_inst_reg <= ex_is_branch_inst;
        ex_mem_is_jump_inst_reg <= ex_is_jump_inst;
        ex_mem_pc_plus_1_reg <= id_ex_pc_plus_1_reg;
        ex_mem_branch_target_reg <= actual_branch_target_calc;
        ex_mem_branch_pc_reg <= ex_branch_pc;
    end
end

// --- MEM Stage ---
assign mem_alu_result = ex_mem_alu_result_reg;
assign mem_rd_addr = ex_mem_rd_addr_reg;
assign mem_reg_write_enable = ex_mem_reg_write_enable_reg;
assign mem_mem_read_enable = ex_mem_mem_read_enable_reg;
assign mem_mem_write_enable = ex_mem_mem_write_enable_reg;
assign mem_mem_addr = ex_mem_alu_result_reg;

// Combinational logic for branch resolution and misprediction detection
assign branch_actual_taken = ex_mem_is_branch_inst_reg && ex_mem_zero_flag_reg;
assign branch_resolved_pc = ex_mem_branch_pc_reg;
assign branch_resolved_target_pc = ex_mem_branch_target_reg;

assign branch_mispredicted_local =
    (ex_mem_is_branch_inst_reg || ex_mem_is_jump_inst_reg) ?
        (ex_mem_is_branch_inst_reg ?
            // Branch instruction misprediction logic
            ((if_btb_predicted_taken != branch_actual_taken) || // Prediction was wrong (taken vs not taken)
             (branch_actual_taken && (if_btb_predicted_next_pc != branch_resolved_target_pc))) : // Taken, but target wrong
            // Jump instruction misprediction logic (only target can be wrong)
            (ex_mem_is_jump_inst_reg && (if_btb_predicted_next_pc != branch_resolved_target_pc))
        ) :
        1'b0; // No branch/jump instruction, so no misprediction

// Sequential registration for branch_mispredicted
always @(posedge clk or posedge reset) begin
    if (reset) begin
        branch_mispredicted <= 1'b0;
    end else begin
        branch_mispredicted <= branch_mispredicted_local; // Register the combinational result
    end
end

// This block is the sole driver for branch_miss_rate_counter
always @(posedge clk or posedge reset) begin
    if (reset) begin
        branch_miss_rate_counter <= 8'h0;
    end else if (branch_mispredicted) begin
        if (branch_miss_rate_counter < 8'hFF) begin
            branch_miss_rate_counter <= branch_miss_rate_counter + 8'h1;
        end
    end else begin
        if (branch_miss_rate_counter > 8'h0) begin
            branch_miss_rate_counter <= branch_miss_rate_counter - 8'h01; // Decay
        end
    end
end

assign debug_branch_miss_rate = branch_miss_rate_counter;

always @(posedge clk or posedge reset) begin
    if (reset || pipeline_flush) begin
        mem_wb_write_data_reg <= 4'h0;
        mem_wb_rd_addr_reg <= 3'h0;
        mem_wb_reg_write_enable_reg <= 1'b0;
    end else if (~pipeline_stall) begin
        if (mem_mem_read_enable) begin
            mem_wb_write_data_reg <= mem_read_data;
        end else begin
            mem_wb_write_data_reg <= mem_alu_result;
        end
        mem_wb_rd_addr_reg <= mem_rd_addr;
        mem_wb_reg_write_enable_reg <= mem_reg_write_enable;
    end
end

// --- WB Stage ---
assign wb_write_data = mem_wb_write_data_reg;
assign wb_rd_addr = mem_wb_rd_addr_reg;
assign wb_reg_write_enable = mem_wb_reg_write_enable_reg;

// --- Debug Outputs ---
assign debug_pc = pc_reg;
assign debug_instr = if_instr;
assign debug_stall = pipeline_stall;
assign debug_flush = pipeline_flush;
assign debug_lock = (new_fsm_control_signal == 2'b11); // FSM state 2'b11 indicates lock
assign debug_fsm_entropy_log = new_fsm_entropy_log;
assign debug_fsm_instr_type_log = new_fsm_instr_type_log;
assign debug_hazard_flag = new_fsm_internal_hazard_flag;
assign debug_fsm_state = new_fsm_control_signal;


endmodule

module archon_top(
input wire clk,
input wire rst_n, // Active low reset for archon_top
input wire [1:0] ml_predicted_action, // ML model's predicted action for AHO and FSM
input wire [7:0] internal_entropy_score, // Entropy score from Quantum Entropy Detector (input to top)
input wire analog_flush_override,
input wire analog_lock_override,
input wire quantum_override_signal,
input wire [15:0] external_entropy_in,


output wire [1:0] fsm_state,
output wire [3:0] debug_pc_out,
output wire [15:0] debug_instr_out,
output wire debug_stall_out,
output wire debug_flush_out,
output wire debug_lock_out,
output wire [7:0] debug_fsm_entropy_log_out,
output wire [2:0] debug_fsm_instr_type_log_out,
output wire debug_hazard_flag // NEW: Expose debug_hazard_flag as an output


);

// Internal wire to connect the reset signal (active high for pipeline_cpu)
wire reset_active_high = ~rst_n;

// The internal_hazard_flag_for_fsm should be derived from internal conditions,
// not an output of archon_top. For this example, let's make it a simple wire
// that could be driven by other internal hazard detection logic if available.
// For now, let's tie it to a dummy value or a simple condition for testing.
// In a real system, this would come from a dedicated hazard detection unit.
wire internal_hazard_flag_for_fsm_internal;
// For demonstration, let's make it high if internal_entropy_score is very high.
// Corrected condition to include 8'd200 itself
assign internal_hazard_flag_for_fsm_internal = (internal_entropy_score >= 8'd200);

pipeline_cpu i_pipeline_cpu (
    .clk(clk),
    .reset(reset_active_high), // Connect active high reset to CPU
    .external_entropy_in(external_entropy_in),
    .ml_predicted_action(ml_predicted_action),
    .internal_hazard_flag_for_fsm(internal_hazard_flag_for_fsm_internal), // Connect the internal hazard flag
    .analog_lock_override_in(analog_lock_override),
    .analog_flush_override_in(analog_flush_override),
    .quantum_override_signal_in(quantum_override_signal),
    .debug_pc(debug_pc_out),
    .debug_instr(debug_instr_out),
    .debug_stall(debug_stall_out),
    .debug_flush(debug_flush_out),
    .debug_lock(debug_lock_out),
    .debug_fsm_entropy_log(debug_fsm_entropy_log_out),
    .debug_fsm_instr_type_log(debug_fsm_instr_type_log_out),
    .debug_hazard_flag(debug_hazard_flag), // Connect debug_hazard_flag output
    .debug_fsm_state(fsm_state)
);

// The internal_entropy_score from the testbench is passed directly to the FSM in pipeline_cpu
// via the internal connection within pipeline_cpu to the qed module and then to fsm_entropy_overlay.
// No direct connection needed here for internal_entropy_score to fsm_entropy_overlay.

endmodule