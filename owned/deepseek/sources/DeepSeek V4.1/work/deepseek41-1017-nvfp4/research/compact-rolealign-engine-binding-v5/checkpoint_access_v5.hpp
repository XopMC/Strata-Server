#pragma once
#include "rolealign_adapter_v5.h"
#include "typed_checkpoint_v5.hpp"
namespace ds1017_rolealign_binding_v5 {
std::shared_ptr<Checkpoint>checkpoint_from_handle(ds1017_rolealign_checkpoint_v5*,bool require_production);
}
