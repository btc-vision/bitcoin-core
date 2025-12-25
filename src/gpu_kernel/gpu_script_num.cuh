// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_SCRIPT_NUM_CUH
#define BITCOIN_GPU_KERNEL_GPU_SCRIPT_NUM_CUH

#include "gpu_script_types.cuh"
#include "gpu_script_stack.cuh"
#include <cstdint>
#include <limits>

namespace gpu {

// ============================================================================
// GPUScriptNum - GPU port of CScriptNum
//
// Numeric opcodes (OP_1ADD, etc) are restricted to operating on 4-byte integers.
// The semantics are subtle: operands must be in the range [-2^31+1...2^31-1],
// but results may overflow (and are valid as long as they are not used in a
// subsequent numeric operation).
//
// GPUScriptNum enforces those semantics by storing results as an int64 and
// returning an error code instead of throwing exceptions.
// ============================================================================

constexpr size_t GPU_DEFAULT_MAX_NUM_SIZE = 4;
constexpr size_t GPU_LOCKTIME_MAX_NUM_SIZE = 5;  // For CHECKLOCKTIMEVERIFY/CHECKSEQUENCEVERIFY

struct GPUScriptNum {
    int64_t m_value;
    bool m_valid;  // Error flag (false = error occurred)

    // ========== Constructors ==========

    __host__ __device__ GPUScriptNum() : m_value(0), m_valid(true) {}

    __host__ __device__ explicit GPUScriptNum(int64_t n) : m_value(n), m_valid(true) {}

    // Construct from stack element with minimal encoding check
    __host__ __device__ GPUScriptNum(const GPUStackElement& elem, bool fRequireMinimal,
                                      size_t nMaxNumSize = GPU_DEFAULT_MAX_NUM_SIZE)
        : m_value(0), m_valid(true)
    {
        if (elem.size > nMaxNumSize) {
            m_valid = false;  // Script number overflow
            return;
        }

        if (fRequireMinimal && elem.size > 0) {
            // Check that the number is encoded with the minimum possible bytes
            // If the most-significant-byte - excluding the sign bit - is zero
            // then we're not minimal. Note how this test also rejects the
            // negative-zero encoding, 0x80.
            if ((elem.data[elem.size - 1] & 0x7f) == 0) {
                // One exception: if there's more than one byte and the most
                // significant bit of the second-most-significant-byte is set
                // it would conflict with the sign bit.
                if (elem.size <= 1 || (elem.data[elem.size - 2] & 0x80) == 0) {
                    m_valid = false;  // Non-minimally encoded script number
                    return;
                }
            }
        }

        m_value = decode(elem.data, elem.size);
    }

    // Construct from raw data
    __host__ __device__ GPUScriptNum(const uint8_t* data, uint16_t size, bool fRequireMinimal,
                                      size_t nMaxNumSize = GPU_DEFAULT_MAX_NUM_SIZE)
        : m_value(0), m_valid(true)
    {
        if (size > nMaxNumSize) {
            m_valid = false;
            return;
        }

        if (fRequireMinimal && size > 0) {
            if ((data[size - 1] & 0x7f) == 0) {
                if (size <= 1 || (data[size - 2] & 0x80) == 0) {
                    m_valid = false;
                    return;
                }
            }
        }

        m_value = decode(data, size);
    }

    // ========== Validity Check ==========

    __host__ __device__ bool IsValid() const { return m_valid; }

    // ========== Comparison Operators ==========

    __host__ __device__ bool operator==(int64_t rhs) const { return m_value == rhs; }
    __host__ __device__ bool operator!=(int64_t rhs) const { return m_value != rhs; }
    __host__ __device__ bool operator<(int64_t rhs) const { return m_value < rhs; }
    __host__ __device__ bool operator<=(int64_t rhs) const { return m_value <= rhs; }
    __host__ __device__ bool operator>(int64_t rhs) const { return m_value > rhs; }
    __host__ __device__ bool operator>=(int64_t rhs) const { return m_value >= rhs; }

    __host__ __device__ bool operator==(const GPUScriptNum& rhs) const { return m_value == rhs.m_value; }
    __host__ __device__ bool operator!=(const GPUScriptNum& rhs) const { return m_value != rhs.m_value; }
    __host__ __device__ bool operator<(const GPUScriptNum& rhs) const { return m_value < rhs.m_value; }
    __host__ __device__ bool operator<=(const GPUScriptNum& rhs) const { return m_value <= rhs.m_value; }
    __host__ __device__ bool operator>(const GPUScriptNum& rhs) const { return m_value > rhs.m_value; }
    __host__ __device__ bool operator>=(const GPUScriptNum& rhs) const { return m_value >= rhs.m_value; }

    // ========== Arithmetic Operators ==========

    __host__ __device__ GPUScriptNum operator+(int64_t rhs) const {
        return GPUScriptNum(m_value + rhs);
    }

    __host__ __device__ GPUScriptNum operator-(int64_t rhs) const {
        return GPUScriptNum(m_value - rhs);
    }

    __host__ __device__ GPUScriptNum operator+(const GPUScriptNum& rhs) const {
        return GPUScriptNum(m_value + rhs.m_value);
    }

    __host__ __device__ GPUScriptNum operator-(const GPUScriptNum& rhs) const {
        return GPUScriptNum(m_value - rhs.m_value);
    }

    __host__ __device__ GPUScriptNum& operator+=(int64_t rhs) {
        m_value += rhs;
        return *this;
    }

    __host__ __device__ GPUScriptNum& operator-=(int64_t rhs) {
        m_value -= rhs;
        return *this;
    }

    __host__ __device__ GPUScriptNum& operator+=(const GPUScriptNum& rhs) {
        m_value += rhs.m_value;
        return *this;
    }

    __host__ __device__ GPUScriptNum& operator-=(const GPUScriptNum& rhs) {
        m_value -= rhs.m_value;
        return *this;
    }

    // Bitwise AND
    __host__ __device__ GPUScriptNum operator&(int64_t rhs) const {
        return GPUScriptNum(m_value & rhs);
    }

    __host__ __device__ GPUScriptNum operator&(const GPUScriptNum& rhs) const {
        return GPUScriptNum(m_value & rhs.m_value);
    }

    __host__ __device__ GPUScriptNum& operator&=(int64_t rhs) {
        m_value &= rhs;
        return *this;
    }

    __host__ __device__ GPUScriptNum& operator&=(const GPUScriptNum& rhs) {
        m_value &= rhs.m_value;
        return *this;
    }

    // Unary negation
    __host__ __device__ GPUScriptNum operator-() const {
        // Note: In CPU version this asserts m_value != INT64_MIN
        // On GPU we just do the operation (undefined behavior for INT64_MIN)
        return GPUScriptNum(-m_value);
    }

    // Assignment
    __host__ __device__ GPUScriptNum& operator=(int64_t rhs) {
        m_value = rhs;
        m_valid = true;
        return *this;
    }

    // ========== Value Retrieval ==========

    // Get as int (clamped to int range)
    __host__ __device__ int32_t getint() const {
        if (m_value > INT32_MAX) return INT32_MAX;
        if (m_value < INT32_MIN) return INT32_MIN;
        return static_cast<int32_t>(m_value);
    }

    // Get as int64
    __host__ __device__ int64_t GetInt64() const { return m_value; }

    // ========== Serialization ==========

    // Serialize value to stack element
    // Returns the number of bytes written (0-9)
    __host__ __device__ uint16_t serialize(uint8_t* dest) const {
        return serialize_value(m_value, dest);
    }

    // Serialize to GPUStackElement
    __host__ __device__ void serialize_to(GPUStackElement& elem) const {
        elem.size = serialize_value(m_value, elem.data);
    }

    // Static serialization function
    __host__ __device__ static uint16_t serialize_value(int64_t value, uint8_t* dest) {
        if (value == 0) {
            return 0;
        }

        const bool neg = value < 0;
        uint64_t absvalue;
        if (neg) {
            // Two's complement conversion for negative values
            absvalue = ~static_cast<uint64_t>(value) + 1;
        } else {
            absvalue = static_cast<uint64_t>(value);
        }

        uint16_t size = 0;
        while (absvalue != 0) {
            dest[size++] = static_cast<uint8_t>(absvalue & 0xff);
            absvalue >>= 8;
        }

        // Handle sign bit
        // - If the most significant byte is >= 0x80 and the value is positive,
        //   push a new zero-byte to make the significant byte < 0x80 again.
        // - If the most significant byte is >= 0x80 and the value is negative,
        //   push a new 0x80 byte that will be popped off when converting to an integral.
        // - If the most significant byte is < 0x80 and the value is negative,
        //   add 0x80 to it, since it will be subtracted and interpreted as a
        //   negative when converting to an integral.
        if (dest[size - 1] & 0x80) {
            dest[size++] = neg ? 0x80 : 0x00;
        } else if (neg) {
            dest[size - 1] |= 0x80;
        }

        return size;
    }

    // ========== Decoding ==========

    // Static decode function (from raw bytes to int64)
    __host__ __device__ static int64_t decode(const uint8_t* data, uint16_t size) {
        if (size == 0) {
            return 0;
        }

        int64_t result = 0;
        for (uint16_t i = 0; i < size; ++i) {
            result |= static_cast<int64_t>(data[i]) << (8 * i);
        }

        // If the input vector's most significant byte is 0x80, remove it from
        // the result's msb and return a negative.
        if (data[size - 1] & 0x80) {
            return -(static_cast<int64_t>(result & ~(0x80ULL << (8 * (size - 1)))));
        }

        return result;
    }
};

// ============================================================================
// Stack-related numeric operations
// ============================================================================

// Pop top element and convert to GPUScriptNum
__host__ __device__ inline bool stack_pop_num(GPUScriptContext* ctx, GPUScriptNum& num,
                                               bool fRequireMinimal,
                                               size_t nMaxNumSize = GPU_DEFAULT_MAX_NUM_SIZE)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUStackElement& elem = stacktop(ctx, -1);
    num = GPUScriptNum(elem, fRequireMinimal, nMaxNumSize);

    if (!num.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size--;
    return true;
}

// Push GPUScriptNum onto stack
__host__ __device__ inline bool stack_push_num(GPUScriptContext* ctx, const GPUScriptNum& num)
{
    if (ctx->stack_size >= MAX_STACK_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
    }

    GPUStackElement& elem = ctx->stack[ctx->stack_size];
    num.serialize_to(elem);
    ctx->stack_size++;
    return true;
}

// ============================================================================
// Opcode Implementations - Arithmetic Operations
// ============================================================================

// OP_1ADD: Add 1 to top
// (n -- n+1)
__host__ __device__ inline bool op_1add(GPUScriptContext* ctx, bool fRequireMinimal)
{
    GPUScriptNum bn(0);
    if (!stack_pop_num(ctx, bn, fRequireMinimal)) return false;
    return stack_push_num(ctx, bn + 1);
}

// OP_1SUB: Subtract 1 from top
// (n -- n-1)
__host__ __device__ inline bool op_1sub(GPUScriptContext* ctx, bool fRequireMinimal)
{
    GPUScriptNum bn(0);
    if (!stack_pop_num(ctx, bn, fRequireMinimal)) return false;
    return stack_push_num(ctx, bn - 1);
}

// OP_NEGATE: Negate top
// (n -- -n)
__host__ __device__ inline bool op_negate(GPUScriptContext* ctx, bool fRequireMinimal)
{
    GPUScriptNum bn(0);
    if (!stack_pop_num(ctx, bn, fRequireMinimal)) return false;
    return stack_push_num(ctx, -bn);
}

// OP_ABS: Absolute value
// (n -- |n|)
__host__ __device__ inline bool op_abs(GPUScriptContext* ctx, bool fRequireMinimal)
{
    GPUScriptNum bn(0);
    if (!stack_pop_num(ctx, bn, fRequireMinimal)) return false;
    if (bn < 0) bn = -bn;
    return stack_push_num(ctx, bn);
}

// OP_NOT: Boolean NOT
// (n -- !n)
__host__ __device__ inline bool op_not(GPUScriptContext* ctx, bool fRequireMinimal)
{
    GPUScriptNum bn(0);
    if (!stack_pop_num(ctx, bn, fRequireMinimal)) return false;
    return stack_push_num(ctx, GPUScriptNum(bn == 0 ? 1 : 0));
}

// OP_0NOTEQUAL: Check if not zero
// (n -- n!=0)
__host__ __device__ inline bool op_0notequal(GPUScriptContext* ctx, bool fRequireMinimal)
{
    GPUScriptNum bn(0);
    if (!stack_pop_num(ctx, bn, fRequireMinimal)) return false;
    return stack_push_num(ctx, GPUScriptNum(bn != 0 ? 1 : 0));
}

// OP_ADD: Add top 2 elements
// (a b -- a+b)
__host__ __device__ inline bool op_add(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, bn1 + bn2);
}

// OP_SUB: Subtract
// (a b -- a-b)
__host__ __device__ inline bool op_sub(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, bn1 - bn2);
}

// OP_BOOLAND: Boolean AND
// (a b -- a&&b)
__host__ __device__ inline bool op_booland(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, GPUScriptNum((bn1 != 0 && bn2 != 0) ? 1 : 0));
}

// OP_BOOLOR: Boolean OR
// (a b -- a||b)
__host__ __device__ inline bool op_boolor(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, GPUScriptNum((bn1 != 0 || bn2 != 0) ? 1 : 0));
}

// OP_NUMEQUAL: Numeric equality
// (a b -- a==b)
__host__ __device__ inline bool op_numequal(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, GPUScriptNum(bn1 == bn2 ? 1 : 0));
}

// OP_NUMEQUALVERIFY: NUMEQUAL then VERIFY
// (a b -- )
__host__ __device__ inline bool op_numequalverify(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (!op_numequal(ctx, fRequireMinimal)) return false;

    GPUScriptNum result(stacktop(ctx, -1), false);
    ctx->stack_size--;

    if (result == 0) {
        return ctx->set_error(GPU_SCRIPT_ERR_NUMEQUALVERIFY);
    }
    return true;
}

// OP_NUMNOTEQUAL: Numeric inequality
// (a b -- a!=b)
__host__ __device__ inline bool op_numnotequal(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, GPUScriptNum(bn1 != bn2 ? 1 : 0));
}

// OP_LESSTHAN: Less than
// (a b -- a<b)
__host__ __device__ inline bool op_lessthan(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, GPUScriptNum(bn1 < bn2 ? 1 : 0));
}

// OP_GREATERTHAN: Greater than
// (a b -- a>b)
__host__ __device__ inline bool op_greaterthan(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, GPUScriptNum(bn1 > bn2 ? 1 : 0));
}

// OP_LESSTHANOREQUAL: Less than or equal
// (a b -- a<=b)
__host__ __device__ inline bool op_lessthanorequal(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, GPUScriptNum(bn1 <= bn2 ? 1 : 0));
}

// OP_GREATERTHANOREQUAL: Greater than or equal
// (a b -- a>=b)
__host__ __device__ inline bool op_greaterthanorequal(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, GPUScriptNum(bn1 >= bn2 ? 1 : 0));
}

// OP_MIN: Minimum of two values
// (a b -- min(a,b))
__host__ __device__ inline bool op_min(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, bn1 < bn2 ? bn1 : bn2);
}

// OP_MAX: Maximum of two values
// (a b -- max(a,b))
__host__ __device__ inline bool op_max(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 2) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -2), fRequireMinimal);
    GPUScriptNum bn2(stacktop(ctx, -1), fRequireMinimal);

    if (!bn1.IsValid() || !bn2.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    ctx->stack_size -= 2;
    return stack_push_num(ctx, bn1 > bn2 ? bn1 : bn2);
}

// OP_WITHIN: Check if value is within range
// (x min max -- min<=x<max)
__host__ __device__ inline bool op_within(GPUScriptContext* ctx, bool fRequireMinimal)
{
    if (ctx->stack_size < 3) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn1(stacktop(ctx, -3), fRequireMinimal);  // x
    GPUScriptNum bn2(stacktop(ctx, -2), fRequireMinimal);  // min
    GPUScriptNum bn3(stacktop(ctx, -1), fRequireMinimal);  // max

    if (!bn1.IsValid() || !bn2.IsValid() || !bn3.IsValid()) {
        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
    }

    bool within = (bn2 <= bn1 && bn1 < bn3);

    ctx->stack_size -= 3;
    return stack_push_num(ctx, GPUScriptNum(within ? 1 : 0));
}

// ============================================================================
// Implementations for forward-declared functions in gpu_script_stack.cuh
// ============================================================================

// OP_DEPTH: Push stack depth
// ( -- stacksize)
__host__ __device__ inline bool op_depth(GPUScriptContext* ctx)
{
    GPUScriptNum bn(ctx->stack_size);
    return stack_push_num(ctx, bn);
}

// OP_SIZE: Push size of top element (without popping)
// (in -- in size)
__host__ __device__ inline bool op_size(GPUScriptContext* ctx)
{
    if (ctx->stack_size < 1) {
        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
    }

    GPUScriptNum bn(stacktop(ctx, -1).size);
    return stack_push_num(ctx, bn);
}

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_SCRIPT_NUM_CUH
