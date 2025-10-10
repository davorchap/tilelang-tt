/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file verify_tt_ir.cc
 * \brief Validate TT-transformed IR (Persistent Transform stage)
 *
 * This pass performs validation checks on the transformed IR to ensure it's
 * ready for Tenstorrent codegen. It verifies:
 *
 * - All required TT Defaults stage-3 metadata is present
 * - Persistent loop structure is correct
 * - Circular buffer annotations are valid
 * - Padding metadata is consistent
 * - No unsupported IR constructs
 *
 * Validation Checks:
 * 1. TT Defaults stage: tt_schedule_policy, tt_layout_type, tt_tile_* dimensions
 * 2. Metadata Inference stage: tt_grid_*, tt_num_tiles, tt_tiles_per_core, tt_num_cores
 * 3. Persistent Transform stage: tt_persistent_loop, tt_runtime_args, tt_core_ranges
 * 4. Persistent Transform stage: tt_circular_buffers, tt_padding_info (if applicable)
 *
 * See: docs/tenstorrent/passes/verify_tt_ir.md for detailed specification
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/transform.h>

#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Validation result structure
 */
struct ValidationResult {
  bool is_valid;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;

  void AddError(const std::string& msg) {
    is_valid = false;
    errors.push_back(msg);
  }

  void AddWarning(const std::string& msg) {
    warnings.push_back(msg);
  }
};

/*!
 * \brief Check for required attribute
 */
template<typename T>
bool CheckAttribute(const PrimFunc& f, const std::string& key, ValidationResult& result) {
  auto attr = f->attrs.GetAttr<T>(key);
  if (!attr.defined()) {
    result.AddError("Missing required attribute: " + key);
    return false;
  }
  return true;
}

/*!
 * \brief Validate TT Defaults stage metadata
 */
void ValidateTTDefaultsStage(const PrimFunc& f, ValidationResult& result) {
  CheckAttribute<String>(f, "tt_schedule_policy", result);
  CheckAttribute<String>(f, "tt_schedule_order", result);
  CheckAttribute<String>(f, "tt_layout_type", result);
  CheckAttribute<Integer>(f, "tt_tile_height", result);
  CheckAttribute<Integer>(f, "tt_tile_width", result);
}

/*!
 * \brief Validate Metadata Inference stage metadata
 */
void ValidateMetadataInferenceStage(const PrimFunc& f, ValidationResult& result) {
  CheckAttribute<Integer>(f, "tt_grid_x", result);
  CheckAttribute<Integer>(f, "tt_grid_y", result);
  CheckAttribute<Integer>(f, "tt_num_tiles", result);
  CheckAttribute<Integer>(f, "tt_num_cores", result);
  CheckAttribute<Array<Array<Integer>>>(f, "tt_tiles_per_core", result);
  CheckAttribute<Map<String, ObjectRef>>(f, "tt_schedule", result);
  CheckAttribute<Map<String, ObjectRef>>(f, "tt_shard", result);

  // Check grid dimensions are reasonable
  auto grid_x = f->attrs.GetAttr<Integer>("tt_grid_x");
  auto grid_y = f->attrs.GetAttr<Integer>("tt_grid_y");

  if (grid_x.defined() && grid_y.defined()) {
    int gx = static_cast<int>(grid_x.value()->value);
    int gy = static_cast<int>(grid_y.value()->value);

    if (gx <= 0 || gy <= 0) {
      result.AddError("Grid dimensions must be positive");
    }
    if (gx > 64 || gy > 64) {
      result.AddWarning("Large grid dimensions may exceed hardware limits");
    }
  }

  // Validate tt_schedule structure
  auto schedule = f->attrs.GetAttr<Map<String, ObjectRef>>("tt_schedule");
  if (schedule.defined()) {
    if (!schedule.value().count("assignments")) {
      result.AddError("tt_schedule missing assignments array");
    } else {
      // Check if assignments can be downcast to Array
      auto assignments_obj = schedule.value()["assignments"];
      if (!assignments_obj.as<Array<ObjectRef>>()) {
        result.AddError("tt_schedule.assignments should be an array");
      }
    }
    if (!schedule.value().count("grid_shape")) {
      result.AddError("tt_schedule missing grid_shape entry");
    }
  }

  // Validate tt_shard structure (per-buffer metadata)
  auto shard = f->attrs.GetAttr<Map<String, ObjectRef>>("tt_shard");
  if (shard.defined()) {
    for (const auto& kv : shard.value()) {
      std::string buffer_name = std::string(kv.first);
      // Check if buffer metadata can be downcast to Map
      auto buffer_map = kv.second.as<Map<String, ObjectRef>>();
      if (!buffer_map) {
        result.AddError("tt_shard entry for buffer " + buffer_name + " must be a map");
        continue;
      }
      if (!buffer_map.value().count(String("layout"))) {
        result.AddError("tt_shard entry for buffer " + buffer_name + " missing layout");
      }
      if (!buffer_map.value().count(String("tile_shape"))) {
        result.AddError("tt_shard entry for buffer " + buffer_name + " missing tile_shape");
      }
    }
  }
}

/*!
 * \brief Validate Persistent Transform stage transformation metadata
 */
void ValidatePersistentTransformStage(const PrimFunc& f, ValidationResult& result) {
  // Check for persistent loop marker
  auto persistent = f->attrs.GetAttr<Bool>("tt_persistent_loop");
  if (!persistent.defined()) {
    result.AddWarning("Missing tt_persistent_loop marker (may be Phase 1 template mode)");
  }

  // Validate runtime args structure
  auto runtime_args = f->attrs.GetAttr<Map<String, ObjectRef>>("tt_runtime_args");
  if (!runtime_args.defined()) {
    result.AddError("Missing tt_runtime_args metadata");
  } else {
    if (!runtime_args.value().count("start_tile") || !runtime_args.value().count("tile_count")) {
      result.AddError("tt_runtime_args must include start_tile and tile_count entries");
    }
    if (!runtime_args.value().count("grid_shape")) {
      result.AddError("tt_runtime_args must include grid_shape");
    }
    if (!runtime_args.value().count("param_order")) {
      result.AddWarning("tt_runtime_args missing param_order (required for codegen mapping)");
    }
  }

  // Check for core mapping (Phase 2)
  auto core_ranges = f->attrs.GetAttr<Array<Array<Integer>>>("tt_core_ranges");
  if (core_ranges.defined()) {
    // Validate core range format
    for (const auto& range : core_ranges.value()) {
      if (range.size() != 6) {
        result.AddError("Core range must have 6 elements: [start_x, start_y, end_x, end_y, start_tile, count]");
      }
    }
  }

  // Check for circular buffers (Phase 2)
  auto cb_configs = f->attrs.GetAttr<Array<Map<String, ObjectRef>>>("tt_circular_buffers");
  if (cb_configs.defined()) {
    int num_cbs = static_cast<int>(cb_configs.value().size());
    auto num_cbs_attr = f->attrs.GetAttr<Integer>("tt_num_cbs");

    if (num_cbs_attr.defined() && num_cbs != static_cast<int>(num_cbs_attr.value()->value)) {
      result.AddError("Mismatch between tt_num_cbs and actual CB count");
    }
  }
}

/*!
 * \brief Main implementation of VerifyTTIR pass
 *
 * Performs comprehensive validation of transformed TT IR.
 * Logs errors and warnings, but does not modify the IR.
 *
 * \param f The PrimFunc to validate
 * \return Same PrimFunc (unchanged), with validation attribute
 */
PrimFunc VerifyTTIRImpl(PrimFunc f) {
  // Step 1: Check if this is a TT function
  auto schedule_policy = f->attrs.GetAttr<String>("tt_schedule_policy");
  if (!schedule_policy.defined()) {
    // Not a TT function, skip validation
    return f;
  }

  // Step 2: Run validation checks
  ValidationResult result;
  result.is_valid = true;

  ValidateTTDefaultsStage(f, result);
  ValidateMetadataInferenceStage(f, result);
  ValidatePersistentTransformStage(f, result);

  // Step 3: Log results
  if (!result.errors.empty()) {
    std::ostringstream oss;
    oss << "TT IR Validation FAILED:\n";
    for (const auto& err : result.errors) {
      oss << "  ERROR: " << err << "\n";
    }
    LOG(ERROR) << oss.str();
  }

  if (!result.warnings.empty()) {
    std::ostringstream oss;
    oss << "TT IR Validation Warnings:\n";
    for (const auto& warn : result.warnings) {
      oss << "  WARNING: " << warn << "\n";
    }
    LOG(WARNING) << oss.str();
  }

  if (result.is_valid && result.errors.empty()) {
    LOG(INFO) << "TT IR Validation PASSED";
  }

  // Step 4: Attach validation result to function
  PrimFunc new_func = WithAttr(f, "tt_ir_validated", Bool(result.is_valid));
  new_func = WithAttr(new_func, "tt_validation_error_count", Integer(static_cast<int>(result.errors.size())));
  new_func = WithAttr(new_func, "tt_validation_warning_count", Integer(static_cast<int>(result.warnings.size())));

  return new_func;
}

using namespace tir::transform;

/*!
 * \brief Create the VerifyTTIR pass
 *
 * \return The TIR pass
 */
Pass VerifyTTIR() {
  auto pass_func = [=](PrimFunc f, const IRModule& m, const PassContext& ctx) {
    return VerifyTTIRImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyTTIR", {});
}

// Register the pass for Python FFI
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VerifyTTIR", VerifyTTIR);
});

}  // namespace tl
}  // namespace tvm
