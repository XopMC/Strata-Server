#include "strata/ds41/native_prefill.hpp"
#include <algorithm>
#include <stdexcept>
namespace strata::ds41 {
NativePrefillLayout native_prefill_layout(int rows,int dim,int inter){
    if(rows<1||rows>16384||dim<32||dim>8192||inter<32||inter>8192||dim%32||inter%32)
        throw std::invalid_argument("native prefill: invalid bounded batch geometry");
    size_t offset=0;
    auto reserve=[&](size_t bytes){offset=(offset+255)&~size_t(255);auto result=offset;offset+=bytes;return result;};
    NativePrefillLayout l;
    l.weight=reserve(size_t(dim)*inter*2);l.x=reserve(size_t(rows)*dim*2);
    l.gate=reserve(size_t(rows)*inter*2);l.up=reserve(size_t(rows)*inter*2);l.hidden=reserve(size_t(rows)*inter*2);
    l.tmp=reserve(size_t(rows)*std::max(dim,inter)*4);l.bytes=(offset+255)&~size_t(255);return l;
}
size_t native_prefill_workspace_bytes(int rows,int dim,int inter){return native_prefill_layout(rows,dim,inter).bytes;}
}
