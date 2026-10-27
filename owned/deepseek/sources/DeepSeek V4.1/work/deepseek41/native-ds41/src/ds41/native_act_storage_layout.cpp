#include "strata/ds41/native_act_storage.hpp"
#include <stdexcept>
namespace strata::ds41 {
NativeActStorageLayout native_act_storage_layout(int rows,int dim){
    if(rows<1||rows>1048576||dim<32||dim>8192||dim%32)throw std::invalid_argument("native activation storage: invalid geometry");
    return {size_t(rows)*dim,size_t(rows)*(dim/32)};
}
}
