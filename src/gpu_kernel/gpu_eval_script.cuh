// Copyright (c) 2024 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_GPU_KERNEL_GPU_EVAL_SCRIPT_CUH
#define BITCOIN_GPU_KERNEL_GPU_EVAL_SCRIPT_CUH

#include "gpu_script_types.cuh"
#include "gpu_script_stack.cuh"
#include "gpu_script_num.cuh"
#include "gpu_opcodes.cuh"
#include "gpu_opcodes_crypto.cuh"
#include "gpu_opcodes_sig.cuh"

namespace gpu {

// ============================================================================
// GPU EvalScript - Main Script Interpreter
// Port of Bitcoin Core's EvalScript from interpreter.cpp
// ============================================================================

// Signature operations are implemented in gpu_opcodes_sig.cuh
// The _impl functions are called directly from the opcode switch

// Main EvalScript function
// Returns true if script executed successfully, false on error
__device__ inline bool EvalScript(
    GPUScriptContext* ctx,
    const uint8_t* script,
    uint32_t script_len)
{
    // Check script size limit (for BASE and WITNESS_V0)
    if ((ctx->sigversion == GPU_SIGVERSION_BASE || ctx->sigversion == GPU_SIGVERSION_WITNESS_V0) &&
        script_len > MAX_SCRIPT_SIZE) {
        return ctx->set_error(GPU_SCRIPT_ERR_SCRIPT_SIZE);
    }

    bool fRequireMinimal = (ctx->verify_flags & GPU_SCRIPT_VERIFY_MINIMALDATA) != 0;

    ctx->pc = 0;
    ctx->opcode_count = 0;

    // Initialize execution data for Tapscript
    if (ctx->sigversion == GPU_SIGVERSION_TAPSCRIPT) {
        ctx->execdata.codeseparator_pos = 0xFFFFFFFF;
        ctx->execdata.codeseparator_pos_init = true;
    }

    // Main execution loop
    while (ctx->pc < script_len) {
        bool fExec = ctx->conditions.all_true();

        // Read opcode
        uint8_t opcode = script[ctx->pc];
        uint32_t opcode_pos = ctx->pc;

        // Handle push data operations
        const uint8_t* push_data = nullptr;
        uint32_t push_size = 0;

        if (opcode <= 0x4b) {
            // Direct push: opcode is the size
            push_size = opcode;
            if (ctx->pc + 1 + push_size > script_len) {
                return ctx->set_error(GPU_SCRIPT_ERR_BAD_OPCODE);
            }
            push_data = &script[ctx->pc + 1];
            ctx->pc += 1 + push_size;
        } else if (opcode == GPU_OP_PUSHDATA1) {
            if (ctx->pc + 2 > script_len) {
                return ctx->set_error(GPU_SCRIPT_ERR_BAD_OPCODE);
            }
            push_size = script[ctx->pc + 1];
            if (ctx->pc + 2 + push_size > script_len) {
                return ctx->set_error(GPU_SCRIPT_ERR_BAD_OPCODE);
            }
            push_data = &script[ctx->pc + 2];
            ctx->pc += 2 + push_size;
        } else if (opcode == GPU_OP_PUSHDATA2) {
            if (ctx->pc + 3 > script_len) {
                return ctx->set_error(GPU_SCRIPT_ERR_BAD_OPCODE);
            }
            push_size = script[ctx->pc + 1] | (static_cast<uint32_t>(script[ctx->pc + 2]) << 8);
            if (ctx->pc + 3 + push_size > script_len) {
                return ctx->set_error(GPU_SCRIPT_ERR_BAD_OPCODE);
            }
            push_data = &script[ctx->pc + 3];
            ctx->pc += 3 + push_size;
        } else if (opcode == GPU_OP_PUSHDATA4) {
            if (ctx->pc + 5 > script_len) {
                return ctx->set_error(GPU_SCRIPT_ERR_BAD_OPCODE);
            }
            push_size = script[ctx->pc + 1] |
                        (static_cast<uint32_t>(script[ctx->pc + 2]) << 8) |
                        (static_cast<uint32_t>(script[ctx->pc + 3]) << 16) |
                        (static_cast<uint32_t>(script[ctx->pc + 4]) << 24);
            if (ctx->pc + 5 + push_size > script_len) {
                return ctx->set_error(GPU_SCRIPT_ERR_BAD_OPCODE);
            }
            push_data = &script[ctx->pc + 5];
            ctx->pc += 5 + push_size;
        } else {
            // Not a push opcode
            ctx->pc++;
        }

        // Check push size limit (520 bytes max for all sigversions)
        // Note: This applies to data being pushed during script execution
        // The tapscript itself can be larger and is handled separately
        if (push_size > MAX_STACK_ELEMENT_SIZE) {
            return ctx->set_error(GPU_SCRIPT_ERR_PUSH_SIZE);
        }

        // Count non-push opcodes (for BASE and WITNESS_V0)
        if (ctx->sigversion == GPU_SIGVERSION_BASE || ctx->sigversion == GPU_SIGVERSION_WITNESS_V0) {
            if (opcode > GPU_OP_16 && ++ctx->opcode_count > MAX_OPS_PER_SCRIPT) {
                return ctx->set_error(GPU_SCRIPT_ERR_OP_COUNT);
            }
        }

        // Check for disabled opcodes
        if (IsOpcodeDisabled(opcode)) {
            return ctx->set_error(GPU_SCRIPT_ERR_DISABLED_OPCODE);
        }

        // Check for OP_CODESEPARATOR restriction
        if (opcode == GPU_OP_CODESEPARATOR && ctx->sigversion == GPU_SIGVERSION_BASE &&
            (ctx->verify_flags & GPU_SCRIPT_VERIFY_CONST_SCRIPTCODE)) {
            return ctx->set_error(GPU_SCRIPT_ERR_OP_CODESEPARATOR);
        }

        // Handle Tapscript OP_SUCCESS
        if (ctx->sigversion == GPU_SIGVERSION_TAPSCRIPT && IsOpcodeSuccess(opcode)) {
            if (ctx->verify_flags & GPU_SCRIPT_VERIFY_DISCOURAGE_OP_SUCCESS) {
                return ctx->set_error(GPU_SCRIPT_ERR_DISCOURAGE_OP_SUCCESS);
            }
            return ctx->set_success();  // Script succeeds immediately
        }

        // Execute push operations
        if (fExec && push_data != nullptr) {
            if (fRequireMinimal && !CheckMinimalPush(push_data, push_size, opcode)) {
                return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
            }
            if (!stack_push(ctx, push_data, push_size)) {
                return false;
            }
        } else if (fExec || IsOpcodeConditional(opcode)) {
            // Execute opcode
            switch (opcode) {
                // ============== Push Value ==============
                case GPU_OP_1NEGATE:
                case GPU_OP_1:
                case GPU_OP_2:
                case GPU_OP_3:
                case GPU_OP_4:
                case GPU_OP_5:
                case GPU_OP_6:
                case GPU_OP_7:
                case GPU_OP_8:
                case GPU_OP_9:
                case GPU_OP_10:
                case GPU_OP_11:
                case GPU_OP_12:
                case GPU_OP_13:
                case GPU_OP_14:
                case GPU_OP_15:
                case GPU_OP_16:
                {
                    GPUScriptNum bn(GetSmallIntegerValue(opcode));
                    if (!stack_push_num(ctx, bn)) return false;
                }
                break;

                // ============== Control Flow ==============
                case GPU_OP_NOP:
                    break;

                case GPU_OP_CHECKLOCKTIMEVERIFY:
                {
                    if (!(ctx->verify_flags & GPU_SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY)) {
                        break;  // Treat as NOP2
                    }
                    if (ctx->stack_size < 1) {
                        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
                    }
                    // Read locktime from stack (5-byte max for locktime)
                    GPUScriptNum nLockTime(stacktop(ctx, -1), fRequireMinimal, GPU_LOCKTIME_MAX_NUM_SIZE);
                    if (!nLockTime.IsValid()) {
                        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
                    }
                    if (nLockTime < 0) {
                        return ctx->set_error(GPU_SCRIPT_ERR_NEGATIVE_LOCKTIME);
                    }
                    // Actual locktime check would be done by signature checker
                    // For now, we just validate the stack operation
                }
                break;

                case GPU_OP_CHECKSEQUENCEVERIFY:
                {
                    if (!(ctx->verify_flags & GPU_SCRIPT_VERIFY_CHECKSEQUENCEVERIFY)) {
                        break;  // Treat as NOP3
                    }
                    if (ctx->stack_size < 1) {
                        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
                    }
                    GPUScriptNum nSequence(stacktop(ctx, -1), fRequireMinimal, GPU_LOCKTIME_MAX_NUM_SIZE);
                    if (!nSequence.IsValid()) {
                        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
                    }
                    if (nSequence < 0) {
                        return ctx->set_error(GPU_SCRIPT_ERR_NEGATIVE_LOCKTIME);
                    }
                    // Check disable flag
                    if ((nSequence.GetInt64() & (1ULL << 31)) != 0) {
                        break;  // Disabled, treat as NOP
                    }
                    // Actual sequence check would be done by signature checker
                }
                break;

                case GPU_OP_NOP1:
                case GPU_OP_NOP4:
                case GPU_OP_NOP5:
                case GPU_OP_NOP6:
                case GPU_OP_NOP7:
                case GPU_OP_NOP8:
                case GPU_OP_NOP9:
                case GPU_OP_NOP10:
                {
                    if (ctx->verify_flags & GPU_SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS) {
                        return ctx->set_error(GPU_SCRIPT_ERR_DISCOURAGE_UPGRADABLE_NOPS);
                    }
                }
                break;

                case GPU_OP_IF:
                case GPU_OP_NOTIF:
                {
                    bool fValue = false;
                    if (!op_if(ctx, opcode == GPU_OP_NOTIF, fExec, fRequireMinimal, ctx->sigversion, fValue)) {
                        return false;
                    }
                }
                break;

                case GPU_OP_ELSE:
                    if (!op_else(ctx)) return false;
                    break;

                case GPU_OP_ENDIF:
                    if (!op_endif(ctx)) return false;
                    break;

                case GPU_OP_VERIFY:
                    if (!op_verify(ctx)) return false;
                    break;

                case GPU_OP_RETURN:
                    return op_return(ctx);

                // ============== Stack Operations ==============
                case GPU_OP_TOALTSTACK:
                    if (!op_toaltstack(ctx)) return false;
                    break;

                case GPU_OP_FROMALTSTACK:
                    if (!op_fromaltstack(ctx)) return false;
                    break;

                case GPU_OP_2DROP:
                    if (!op_2drop(ctx)) return false;
                    break;

                case GPU_OP_2DUP:
                    if (!op_2dup(ctx)) return false;
                    break;

                case GPU_OP_3DUP:
                    if (!op_3dup(ctx)) return false;
                    break;

                case GPU_OP_2OVER:
                    if (!op_2over(ctx)) return false;
                    break;

                case GPU_OP_2ROT:
                    if (!op_2rot(ctx)) return false;
                    break;

                case GPU_OP_2SWAP:
                    if (!op_2swap(ctx)) return false;
                    break;

                case GPU_OP_IFDUP:
                    if (!op_ifdup(ctx)) return false;
                    break;

                case GPU_OP_DEPTH:
                    if (!op_depth(ctx)) return false;
                    break;

                case GPU_OP_DROP:
                    if (!op_drop(ctx)) return false;
                    break;

                case GPU_OP_DUP:
                    if (!op_dup(ctx)) return false;
                    break;

                case GPU_OP_NIP:
                    if (!op_nip(ctx)) return false;
                    break;

                case GPU_OP_OVER:
                    if (!op_over(ctx)) return false;
                    break;

                case GPU_OP_PICK:
                case GPU_OP_ROLL:
                {
                    if (ctx->stack_size < 2) {
                        return ctx->set_error(GPU_SCRIPT_ERR_INVALID_STACK_OPERATION);
                    }
                    GPUScriptNum bn(stacktop(ctx, -1), fRequireMinimal);
                    if (!bn.IsValid()) {
                        return ctx->set_error(GPU_SCRIPT_ERR_MINIMALDATA);
                    }
                    int32_t n = bn.getint();
                    ctx->stack_size--;  // Pop the index
                    if (opcode == GPU_OP_PICK) {
                        if (!op_pick(ctx, n)) return false;
                    } else {
                        if (!op_roll(ctx, n)) return false;
                    }
                }
                break;

                case GPU_OP_ROT:
                    if (!op_rot(ctx)) return false;
                    break;

                case GPU_OP_SWAP:
                    if (!op_swap(ctx)) return false;
                    break;

                case GPU_OP_TUCK:
                    if (!op_tuck(ctx)) return false;
                    break;

                // ============== Splice Operations ==============
                case GPU_OP_SIZE:
                    if (!op_size(ctx)) return false;
                    break;

                // ============== Bitwise Logic ==============
                case GPU_OP_EQUAL:
                    if (!op_equal(ctx)) return false;
                    break;

                case GPU_OP_EQUALVERIFY:
                    if (!op_equalverify(ctx)) return false;
                    break;

                // ============== Arithmetic ==============
                case GPU_OP_1ADD:
                    if (!op_1add(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_1SUB:
                    if (!op_1sub(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_NEGATE:
                    if (!op_negate(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_ABS:
                    if (!op_abs(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_NOT:
                    if (!op_not(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_0NOTEQUAL:
                    if (!op_0notequal(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_ADD:
                    if (!op_add(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_SUB:
                    if (!op_sub(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_BOOLAND:
                    if (!op_booland(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_BOOLOR:
                    if (!op_boolor(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_NUMEQUAL:
                    if (!op_numequal(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_NUMEQUALVERIFY:
                    if (!op_numequalverify(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_NUMNOTEQUAL:
                    if (!op_numnotequal(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_LESSTHAN:
                    if (!op_lessthan(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_GREATERTHAN:
                    if (!op_greaterthan(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_LESSTHANOREQUAL:
                    if (!op_lessthanorequal(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_GREATERTHANOREQUAL:
                    if (!op_greaterthanorequal(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_MIN:
                    if (!op_min(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_MAX:
                    if (!op_max(ctx, fRequireMinimal)) return false;
                    break;

                case GPU_OP_WITHIN:
                    if (!op_within(ctx, fRequireMinimal)) return false;
                    break;

                // ============== Crypto ==============
                case GPU_OP_RIPEMD160:
                    if (!op_ripemd160(ctx)) return false;
                    break;

                case GPU_OP_SHA1:
                    if (!op_sha1(ctx)) return false;
                    break;

                case GPU_OP_SHA256:
                    if (!op_sha256(ctx)) return false;
                    break;

                case GPU_OP_HASH160:
                    if (!op_hash160(ctx)) return false;
                    break;

                case GPU_OP_HASH256:
                    if (!op_hash256(ctx)) return false;
                    break;

                case GPU_OP_CODESEPARATOR:
                    if (!op_codeseparator(ctx, opcode_pos)) return false;
                    break;

                // Signature operations - implemented in gpu_opcodes_sig.cuh
                case GPU_OP_CHECKSIG:
                    if (!op_checksig_impl(ctx, script, script_len)) return false;
                    break;

                case GPU_OP_CHECKSIGVERIFY:
                    if (!op_checksigverify_impl(ctx, script, script_len)) return false;
                    break;

                case GPU_OP_CHECKMULTISIG:
                    if (!op_checkmultisig_impl(ctx, script, script_len, fRequireMinimal)) return false;
                    break;

                case GPU_OP_CHECKMULTISIGVERIFY:
                    if (!op_checkmultisigverify_impl(ctx, script, script_len, fRequireMinimal)) return false;
                    break;

                case GPU_OP_CHECKSIGADD:
                    if (!op_checksigadd_impl(ctx, script, script_len)) return false;
                    break;

                default:
                    return ctx->set_error(GPU_SCRIPT_ERR_BAD_OPCODE);
            }
        }

        // Check combined stack size
        if (static_cast<uint32_t>(ctx->stack_size + ctx->altstack_size) > MAX_STACK_SIZE) {
            return ctx->set_error(GPU_SCRIPT_ERR_STACK_SIZE);
        }
    }

    // Check for unbalanced conditionals
    if (!ctx->conditions.empty()) {
        return ctx->set_error(GPU_SCRIPT_ERR_UNBALANCED_CONDITIONAL);
    }

    return ctx->set_success();
}

} // namespace gpu

#endif // BITCOIN_GPU_KERNEL_GPU_EVAL_SCRIPT_CUH
