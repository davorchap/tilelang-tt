#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {
namespace tt {

using namespace tir;

constexpr CallEffectKind kOpaque = CallEffectKind::kOpaque;

#define TT_REGISTER_OP(Name, NumInputs)                                        \
  TVM_REGISTER_OP(Name)                                                        \
      .set_num_inputs(NumInputs)                                               \
      .set_attr<TCallEffectKind>("TCallEffectKind", Integer(kOpaque))          \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", Name)

// Matmul lifecycle intrinsics
TT_REGISTER_OP("tt.mm_init", 3);
TT_REGISTER_OP("tt.matmul_tiles", 6);
TT_REGISTER_OP("tt.tile_regs_acquire", 0);
TT_REGISTER_OP("tt.tile_regs_release", 0);
TT_REGISTER_OP("tt.tile_regs_commit", 0);
TT_REGISTER_OP("tt.tile_regs_wait", 0);
TT_REGISTER_OP("tt.pack_tile", 2);

// Circular buffer intrinsics
TT_REGISTER_OP("tt.cb_wait_front", 2);
TT_REGISTER_OP("tt.cb_pop_front", 2);
TT_REGISTER_OP("tt.cb_reserve_back", 2);
TT_REGISTER_OP("tt.cb_push_back", 2);

// Elementwise helpers
TT_REGISTER_OP("tt.binary_op_init_common", 3);
TT_REGISTER_OP("tt.add_tiles_init", 0);
TT_REGISTER_OP("tt.add_tiles", 5);
TT_REGISTER_OP("tt.mul_tiles_init", 0);
TT_REGISTER_OP("tt.mul_tiles", 5);

// Layout helpers
TT_REGISTER_OP("tt.tilize", 3);
TT_REGISTER_OP("tt.untilize", 3);

#undef TT_REGISTER_OP

} // namespace tt
} // namespace tl
} // namespace tvm
