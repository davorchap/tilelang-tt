#pragma once

#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace tl {
namespace tt {

/*!
 * \brief Helper to build a call to a TT intrinsic Op.
 * \param op_name Name of the intrinsic (must be registered via TVM_REGISTER_OP).
 * \param args Arguments passed to the intrinsic call.
 * \return TIR statement wrapping the intrinsic call.
 */
inline tir::Stmt EvaluateIntrinsic(const char* op_name, tir::Array<tir::PrimExpr> args) {
  tir::Op op = tir::Op::Get(op_name);
  tir::PrimExpr call =
      tir::Call(tir::DataType::Void(), op, std::move(args));
  return tir::Evaluate(std::move(call));
}

/*!
 * \brief Helper to build a PrimExpr TT intrinsic call (without Evaluate wrapper).
 * \param op_name Name of the intrinsic (must be registered via TVM_REGISTER_OP).
 * \param args Arguments passed to the intrinsic call.
 * \return PrimExpr representing the intrinsic call.
 */
inline tir::PrimExpr CallIntrinsic(const char* op_name, tir::Array<tir::PrimExpr> args,
                                   tir::DataType dtype = tir::DataType::Void()) {
  tir::Op op = tir::Op::Get(op_name);
  return tir::Call(dtype, op, std::move(args));
}

}  // namespace tt
}  // namespace tl
}  // namespace tvm

