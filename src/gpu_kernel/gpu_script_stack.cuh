// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_SCRIPT_STACK_CUH
#define BITCOIN_GPU_KERNEL_GPU_SCRIPT_STACK_CUH

#include "gpu_script_types.cuh"
#include <cstring>

namespace gpu {

// ============================================================================
// CastToBool - Convert stack element to boolean
// Matches Bitcoin Core's CastToBool exactly (interpreter.cpp:36-49)
// ============================================================================

__host__ __device__ inline bool CastToBool(const GPUStackElement& elem)
{
    for (uint16_t i = 0; i < elem.size; i++) {
        if (elem.data[i] != 0) {
            // Can be negative zero
            if (i == elem.size - 1 && elem.data[i] == 0x80)
                return false;
            return true;
        }
    }
    return false;
}

__host__ __device__ inline bool CastToBool(const uint8_t* data, uint16_t size)
{
    for (uint16_t i = 0; i < size; i++) {
        if (data[i] != 0) {
            // Can be negative zero
            if (i == size - 1 && data[i] == 0x80)
                return false;
            return true;
        }
    }
    return false;
}

// ============================================================================
// Stack Access Helpers
// ============================================================================

// Access element from top of stack (negative index, -1 = top)
__host__ __device__ inline GPUStackElement& stacktop(GPUScriptContext* ctx, int32_t idx)
{
    return ctx->stack[ctx->stack_size + idx];
}

__host__ __device__ inline const GPUStackElement& stacktop(const GPUScriptContext* ctx, int32_t idx)
{
    return ctx->stack[ctx->stack_size + idx];
}

// Access element from top of altstack
__host__ __device__ inline GPUStackElement& altstacktop(GPUScriptContext* ctx, int32_t idx)
{
    return ctx->altstack[ctx->altstack_size + idx];
}

__host__ __device__ inline const GPUStackElement& altstacktop(const GPUScriptContext* ctx, int32_t idx)
{
    return ctx->altstack[ctx->altstack_size + idx];
}

// ============================================================================
// Basic Stack Operations
// ============================================================================

// Push data onto stack
// 520-byte limit applies to ALL sigversions for stack elements
// (The tapscript itself is handled separately via tapscript_buffer)
__host__ __device__ inline bool stack_push(GPUScriptContext* ctx, const uint8_t* data, uint16_t size)
{
    if (ctx->stack_size >= MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }

    // 520-byte limit for stack elements (applies to all sigversions)
    if (size > MAX_STACK_ELEMENT_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_PUSH_SIZE);
    }

    GPUStackElement& elem = ctx->stack[ctx->stack_size];
    if (size > 0) {
        memcpy(elem.data, data, size);
    }
    elem.size = size;
    ctx->stack_size++;
    return true;
}

// Push a single byte onto stack
__host__ __device__ inline bool stack_push_byte(GPUScriptContext* ctx, uint8_t value)
{
    return stack_push(ctx, &value, 1);
}

// Push empty element onto stack
__host__ __device__ inline bool stack_push_empty(GPUScriptContext* ctx)
{
    if (ctx->stack_size >= MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }
    ctx->stack[ctx->stack_size].size = 0;
    ctx->stack_size++;
    return true;
}

// Push boolean onto stack (true = [0x01], false = [])
__host__ __device__ inline bool stack_push_bool(GPUScriptContext* ctx, bool value)
{
    if (value) {
        uint8_t one = 0x01;
        return stack_push(ctx, &one, 1);
    } else {
        return stack_push_empty(ctx);
    }
}

// Pop element from stack (just decrements size, data remains)
__host__ __device__ inline bool stack_pop(GPUScriptContext* ctx)
{
    if (ctx->stack_size == 0) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    ctx->stack_size--;
    return true;
}

// Pop element and copy to destination
__host__ __device__ inline bool stack_pop_to(GPUScriptContext* ctx, GPUStackElement& dest)
{
    if (ctx->stack_size == 0) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    ctx->stack_size--;
    memcpy(dest.data, ctx->stack[ctx->stack_size].data, ctx->stack[ctx->stack_size].size);
    dest.size = ctx->stack[ctx->stack_size].size;
    return true;
}

// Copy element from stack to destination (without popping)
__host__ __device__ inline void stack_copy_element(const GPUStackElement& src, GPUStackElement& dest)
{
    memcpy(dest.data, src.data, src.size);
    dest.size = src.size;
}

// Swap two stack elements
__host__ __device__ inline void stack_swap_elements(GPUStackElement& a, GPUStackElement& b)
{
    // Swap sizes
    uint16_t tmp_size = a.size;
    a.size = b.size;
    b.size = tmp_size;

    // Swap data (use max of both sizes to ensure complete swap)
    uint16_t max_size = (a.size > b.size) ? a.size : b.size;
    max_size = (max_size > tmp_size) ? max_size : tmp_size;

    for (uint16_t i = 0; i < max_size; i++) {
        uint8_t tmp = a.data[i];
        a.data[i] = b.data[i];
        b.data[i] = tmp;
    }
}

// Compare two stack elements for equality
__host__ __device__ inline bool stack_elements_equal(const GPUStackElement& a, const GPUStackElement& b)
{
    if (a.size != b.size) return false;
    for (uint16_t i = 0; i < a.size; i++) {
        if (a.data[i] != b.data[i]) return false;
    }
    return true;
}

// ============================================================================
// Altstack Operations
// ============================================================================

// Push to altstack
__host__ __device__ inline bool altstack_push(GPUScriptContext* ctx, const GPUStackElement& elem)
{
    if (ctx->altstack_size >= MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }
    stack_copy_element(elem, ctx->altstack[ctx->altstack_size]);
    ctx->altstack_size++;
    return true;
}

// Pop from altstack
__host__ __device__ inline bool altstack_pop(GPUScriptContext* ctx)
{
    if (ctx->altstack_size == 0) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_ALTSTACK_OPERATION);
    }
    ctx->altstack_size--;
    return true;
}

// ============================================================================
// Opcode Implementations - Stack Operations
// ============================================================================

// OP_TOALTSTACK: Move top element from main stack to altstack
// (x -- ) altstack: ( -- x)
__host__ __device__ inline bool op_toaltstack(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    if (!altstack_push(ctx, stacktop(ctx, -1))) {
        return false;
    }
    return stack_pop(ctx);
}

// OP_FROMALTSTACK: Move top element from altstack to main stack
// ( -- x) altstack: (x -- )
__host__ __device__ inline bool op_fromaltstack(GPUScriptContext* ctx)
{
    if (ctx->altstack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_ALTSTACK_OPERATION);
    }

    GPUStackElement& elem = altstacktop(ctx, -1);
    if (!stack_push(ctx, elem.data, elem.size)) {
        return false;
    }
    return altstack_pop(ctx);
}

// OP_2DROP: Drop top 2 elements
// (x1 x2 -- )
__host__ __device__ inline bool op_2drop(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    ctx->stack_size -= 2;
    return true;
}

// OP_2DUP: Duplicate top 2 elements
// (x1 x2 -- x1 x2 x1 x2)
__host__ __device__ inline bool op_2dup(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    if (static_cast<uint32_t>(ctx->stack_size + 2) > MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }

    GPUStackElement& vch1 = stacktop(ctx, -2);
    GPUStackElement& vch2 = stacktop(ctx, -1);

    stack_copy_element(vch1, ctx->stack[ctx->stack_size]);
    ctx->stack_size++;
    stack_copy_element(vch2, ctx->stack[ctx->stack_size]);
    ctx->stack_size++;

    return true;
}

// OP_3DUP: Duplicate top 3 elements
// (x1 x2 x3 -- x1 x2 x3 x1 x2 x3)
__host__ __device__ inline bool op_3dup(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 3) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    if (static_cast<uint32_t>(ctx->stack_size + 3) > MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }

    GPUStackElement& vch1 = stacktop(ctx, -3);
    GPUStackElement& vch2 = stacktop(ctx, -2);
    GPUStackElement& vch3 = stacktop(ctx, -1);

    stack_copy_element(vch1, ctx->stack[ctx->stack_size]);
    ctx->stack_size++;
    stack_copy_element(vch2, ctx->stack[ctx->stack_size]);
    ctx->stack_size++;
    stack_copy_element(vch3, ctx->stack[ctx->stack_size]);
    ctx->stack_size++;

    return true;
}

// OP_2OVER: Copy 2 elements from 4 back
// (x1 x2 x3 x4 -- x1 x2 x3 x4 x1 x2)
__host__ __device__ inline bool op_2over(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 4) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    if (static_cast<uint32_t>(ctx->stack_size + 2) > MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }

    GPUStackElement& vch1 = stacktop(ctx, -4);
    GPUStackElement& vch2 = stacktop(ctx, -3);

    stack_copy_element(vch1, ctx->stack[ctx->stack_size]);
    ctx->stack_size++;
    stack_copy_element(vch2, ctx->stack[ctx->stack_size]);
    ctx->stack_size++;

    return true;
}

// OP_2ROT: Rotate 6 elements - move bottom 2 to top
// (x1 x2 x3 x4 x5 x6 -- x3 x4 x5 x6 x1 x2)
__host__ __device__ inline bool op_2rot(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 6) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    // Save x1 and x2
    GPUStackElement tmp1, tmp2;
    stack_copy_element(stacktop(ctx, -6), tmp1);
    stack_copy_element(stacktop(ctx, -5), tmp2);

    // Shift x3-x6 down (erase positions -6 and -5)
    for (uint16_t i = ctx->stack_size - 6; i < ctx->stack_size - 2; i++) {
        stack_copy_element(ctx->stack[i + 2], ctx->stack[i]);
    }

    // Place x1 and x2 at top
    stack_copy_element(tmp1, ctx->stack[ctx->stack_size - 2]);
    stack_copy_element(tmp2, ctx->stack[ctx->stack_size - 1]);

    return true;
}

// OP_2SWAP: Swap top 2 pairs
// (x1 x2 x3 x4 -- x3 x4 x1 x2)
__host__ __device__ inline bool op_2swap(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 4) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    stack_swap_elements(stacktop(ctx, -4), stacktop(ctx, -2));
    stack_swap_elements(stacktop(ctx, -3), stacktop(ctx, -1));

    return true;
}

// OP_IFDUP: Duplicate top if not zero
// (x -- x) or (x -- x x)
__host__ __device__ inline bool op_ifdup(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUStackElement& vch = stacktop(ctx, -1);
    if (CastToBool(vch)) {
        if (ctx->stack_size >= MAX_STACK_SIZE) {
            return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
        }
        stack_copy_element(vch, ctx->stack[ctx->stack_size]);
        ctx->stack_size++;
    }

    return true;
}

// OP_DEPTH: Push stack depth
// ( -- stacksize)
__host__ __device__ inline bool op_depth(GPUScriptContext* ctx);  // Forward declaration (needs CScriptNum)

// OP_DROP: Drop top element
// (x -- )
__host__ __device__ inline bool op_drop(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    ctx->stack_size--;
    return true;
}

// OP_DUP: Duplicate top element
// (x -- x x)
__host__ __device__ inline bool op_dup(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    if (ctx->stack_size >= MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }

    stack_copy_element(stacktop(ctx, -1), ctx->stack[ctx->stack_size]);
    ctx->stack_size++;

    return true;
}

// OP_NIP: Remove second-to-top element
// (x1 x2 -- x2)
__host__ __device__ inline bool op_nip(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    // Move top element down one position
    stack_copy_element(stacktop(ctx, -1), stacktop(ctx, -2));
    ctx->stack_size--;

    return true;
}

// OP_OVER: Copy second-to-top to top
// (x1 x2 -- x1 x2 x1)
__host__ __device__ inline bool op_over(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    if (ctx->stack_size >= MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }

    stack_copy_element(stacktop(ctx, -2), ctx->stack[ctx->stack_size]);
    ctx->stack_size++;

    return true;
}

// OP_PICK: Copy element n positions back to top
// (xn ... x2 x1 x0 n -- xn ... x2 x1 x0 xn)
__host__ __device__ inline bool op_pick(GPUScriptContext* ctx, int32_t n)
{
    if (n < 0 || n >= (int32_t)ctx->stack_size) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    if (ctx->stack_size >= MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }

    stack_copy_element(stacktop(ctx, -n - 1), ctx->stack[ctx->stack_size]);
    ctx->stack_size++;

    return true;
}

// OP_ROLL: Move element n positions back to top
// (xn ... x2 x1 x0 n -- ... x2 x1 x0 xn)
__host__ __device__ inline bool op_roll(GPUScriptContext* ctx, int32_t n)
{
    if (n < 0 || n >= (int32_t)ctx->stack_size) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    if (n == 0) {
        // No-op for n=0
        return true;
    }

    // Save the element at position -n-1
    GPUStackElement tmp;
    stack_copy_element(stacktop(ctx, -n - 1), tmp);

    // Shift elements down
    uint16_t start_idx = ctx->stack_size - n - 1;
    for (uint16_t i = start_idx; i < ctx->stack_size - 1; i++) {
        stack_copy_element(ctx->stack[i + 1], ctx->stack[i]);
    }

    // Place saved element at top
    stack_copy_element(tmp, ctx->stack[ctx->stack_size - 1]);

    return true;
}

// OP_ROT: Rotate top 3 elements
// (x1 x2 x3 -- x2 x3 x1)
__host__ __device__ inline bool op_rot(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 3) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    // x2 x1 x3 after first swap
    stack_swap_elements(stacktop(ctx, -3), stacktop(ctx, -2));
    // x2 x3 x1 after second swap
    stack_swap_elements(stacktop(ctx, -2), stacktop(ctx, -1));

    return true;
}

// OP_SWAP: Swap top 2 elements
// (x1 x2 -- x2 x1)
__host__ __device__ inline bool op_swap(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    stack_swap_elements(stacktop(ctx, -2), stacktop(ctx, -1));

    return true;
}

// OP_TUCK: Copy top to before second-to-top
// (x1 x2 -- x2 x1 x2)
__host__ __device__ inline bool op_tuck(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }
    if (ctx->stack_size >= MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }

    // Save top element
    GPUStackElement tmp;
    stack_copy_element(stacktop(ctx, -1), tmp);

    // Make room by shifting last 2 elements up
    stack_copy_element(ctx->stack[ctx->stack_size - 1], ctx->stack[ctx->stack_size]);
    stack_copy_element(ctx->stack[ctx->stack_size - 2], ctx->stack[ctx->stack_size - 1]);

    // Insert copy at position -2 (now at stack_size - 2)
    stack_copy_element(tmp, ctx->stack[ctx->stack_size - 2]);

    ctx->stack_size++;

    return true;
}

// OP_SIZE: Push size of top element (without popping)
// (in -- in size)
__host__ __device__ inline bool op_size(GPUScriptContext* ctx);  // Forward declaration (needs CScriptNum)

// ============================================================================
// Splice Operations (mostly disabled, but SIZE is needed)
// ============================================================================

// OP_CAT, OP_SUBSTR, OP_LEFT, OP_RIGHT are disabled (CVE-2010-5137)
// Only OP_SIZE is enabled

// ============================================================================
// Bitwise Logic Operations
// ============================================================================

// OP_EQUAL: Check if top 2 elements are equal
// (x1 x2 -- bool)
__host__ __device__ inline bool op_equal(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUStackElement& vch1 = stacktop(ctx, -2);
    GPUStackElement& vch2 = stacktop(ctx, -1);

    bool equal = stack_elements_equal(vch1, vch2);

    // Pop both elements
    ctx->stack_size -= 2;

    // Push result
    return stack_push_bool(ctx, equal);
}

// OP_EQUALVERIFY: EQUAL then VERIFY
// (x1 x2 -- ) or fail
__host__ __device__ inline bool op_equalverify(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUStackElement& vch1 = stacktop(ctx, -2);
    GPUStackElement& vch2 = stacktop(ctx, -1);

    bool equal = stack_elements_equal(vch1, vch2);

    // Pop both elements
    ctx->stack_size -= 2;

    if (!equal) {
        return ctx->set_error(GPU_SCRIPT_ERR_EQUALVERIFY);
    }

    return true;
}

// ============================================================================
// Control Flow Operations
// ============================================================================

// OP_VERIFY: Fail if top is false, pop if true
// (true -- ) or (false -- false) and fail
__host__ __device__ inline bool op_verify(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    bool value = CastToBool(stacktop(ctx, -1));
    if (value) {
        ctx->stack_size--;
        return true;
    } else {
        return ctx->set_error(GPU_SCRIPT_ERR_VERIFY);
    }
}

// OP_RETURN: Always fails
__host__ __device__ inline bool op_return(GPUScriptContext* ctx)
{
    return ctx->set_error(GPU_SCRIPT_ERR_OP_RETURN);
}

// OP_IF/OP_NOTIF: Conditional execution
// Returns the value to push to condition stack (after handling NOTIF inversion)
__host__ __device__ inline bool op_if(GPUScriptContext* ctx, bool is_notif, bool fExec,
                                       bool require_minimal, GPUSigVersion sigversion, bool& result)
{
    result = false;

    if (fExec) {
        if (ctx->stack_size < 1) {
            return ctx->set_error(GPU_SCRIPT_ERR_UNBALANCED_CONDITIONAL);
        }

        GPUStackElement& vch = stacktop(ctx, -1);

        // Tapscript requires minimal IF/NOTIF inputs
        if (sigversion == GPU_SIGVERSION_TAPSCRIPT) {
            if (vch.size > 1 || (vch.size == 1 && vch.data[0] != 1)) {
                return ctx->set_error(GPU_SCRIPT_ERR_TAPSCRIPT_MINIMALIF);
            }
        }

        // WITNESS_V0 with MINIMALIF flag
        if (sigversion == GPU_SIGVERSION_WITNESS_V0 &&
            (ctx->verify_flags & GPU_SCRIPT_VERIFY_MINIMALIF)) {
            if (vch.size > 1) {
                return ctx->set_error(GPU_SCRIPT_ERR_MINIMALIF);
            }
            if (vch.size == 1 && vch.data[0] != 1) {
                return ctx->set_error(GPU_SCRIPT_ERR_MINIMALIF);
            }
        }

        result = CastToBool(vch);
        if (is_notif) {
            result = !result;
        }

        ctx->stack_size--;
    }

    ctx->conditions.push_back(result);
    return true;
}

// OP_ELSE: Toggle top condition
__host__ __device__ inline bool op_else(GPUScriptContext* ctx)
{
    if (ctx->conditions.empty()) {
        return ctx->set_error(GPU_SCRIPT_ERR_UNBALANCED_CONDITIONAL);
    }
    ctx->conditions.toggle_top();
    return true;
}

// OP_ENDIF: End conditional
__host__ __device__ inline bool op_endif(GPUScriptContext* ctx)
{
    if (ctx->conditions.empty()) {
        return ctx->set_error(GPU_SCRIPT_ERR_UNBALANCED_CONDITIONAL);
    }
    ctx->conditions.pop_back();
    return true;
}

// ============================================================================
// Stack Size Checking
// ============================================================================

// Check total stack size limit (main + alt)
__host__ __device__ inline bool check_stack_size(GPUScriptContext* ctx)
{
    if (ctx->stack_size + ctx->altstack_size > MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }
    return true;
}

// ============================================================================
// Helper: Copy data between raw pointers and stack elements
// ============================================================================

__host__ __device__ inline void copy_to_element(GPUStackElement& elem, const uint8_t* data, uint16_t size)
{
    if (size > MAX_STACK_ELEMENT_SIZE) size = MAX_STACK_ELEMENT_SIZE;
    memcpy(elem.data, data, size);
    elem.size = size;
}

__host__ __device__ inline void copy_from_element(uint8_t* dest, const GPUStackElement& elem)
{
    memcpy(dest, elem.data, elem.size);
}

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_SCRIPT_STACK_CUH
