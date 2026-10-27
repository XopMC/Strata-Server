#include "strata/ds41/native_adapter_compact_v248.h"
#include "strata/ds41/native_experts.hpp"
#include "strata/ds41/native_checkpoint.hpp"
#include <algorithm>
#include <stdexcept>
#include "pair_registry_v145.hpp"
using namespace strata::ds41;
namespace {
thread_local std::string error;
using Registry=std::shared_ptr<BorrowedWeights>;
void need(bool ok){if(!ok)throw std::invalid_argument("invalid native adapter arguments");}
template<class F>int checked(F fn){try{error.clear();fn();return 0;}catch(const std::exception&e){error=e.what();return 1;}catch(...){error="unknown native adapter exception";return 2;}}
}
extern "C" {
int ds41_native_adapter_abi(void){return 1;}
const char* ds41_native_adapter_error(void){return error.c_str();}
void* ds41_registry_create(const ds41_memory_region* regions,size_t count,void* token,ds41_owner_callback retain,ds41_owner_callback release){
    void* result=nullptr;
    checked([&]{
        need(regions&&count&&count<=100000&&token&&retain&&release);
        std::vector<MemoryRegion> values;values.reserve(count);
        for(size_t i=0;i<count;i++)values.push_back({regions[i].data,regions[i].bytes});
        retain(token);std::shared_ptr<const void> owner(token,[release](const void*p){release(const_cast<void*>(p));});
        result=new Registry(std::make_shared<BorrowedWeights>(std::move(owner),std::move(values)));
    });return result;
}
int ds41_registry_add(void* registry,const ds41_tensor_view* view){return checked([&]{
    need(registry&&view&&view->name&&view->rank<=8&&(view->rank==0||view->shape));
    need(view->encoding>=0&&view->encoding<=int(TensorEncoding::Uniform));
    need(view->protected_native_rows==0||view->protected_native_rows==1);
    std::vector<int64_t>shape;if(view->rank)shape.assign(view->shape,view->shape+view->rank);
    (*static_cast<Registry*>(registry))->add(view->name,{TensorEncoding(view->encoding),std::move(shape),view->data,view->bytes,view->bits,bool(view->protected_native_rows)});
});}
int ds41_registry_add_output16_compact_v248(void*registry,const char*name,const void*data,size_t bytes,int rows,int columns,int bits){return checked([&]{
    need(registry&&name&&data&&bytes&&rows>=16&&rows%16==0&&columns>=32&&columns%32==0&&(bits==0||bits==3));
    (*static_cast<Registry*>(registry))->add(name,{TensorEncoding::Output16Compact,{rows,columns},static_cast<const uint8_t*>(data),bytes,bits,false});
});}
int ds41_registry_output16_info_v248(void*registry,uint64_t*out,size_t count){return checked([&]{
    need(registry&&out&&count==3);const auto values=(*static_cast<Registry*>(registry))->output16_info_v248();
    std::copy(values.begin(),values.end(),out);
});}
int ds145_registry_add_pair32(void*registry,const char*name,const Ds143PairDescriptor*d){return checked([&]{
    need(registry&&name&&d);pair145::add(*static_cast<Registry*>(registry),name,*d);
});}
int ds145_registry_pair32_info(void*registry,uint64_t*out,size_t count){return checked([&]{
    need(registry&&out&&count==8);const auto v=pair145::info(**static_cast<Registry*>(registry));std::copy(v.begin(),v.end(),out);
});}
int ds145_registry_pair32_upload_info(void*registry,uint64_t*out,size_t count){return checked([&]{
    need(registry&&out&&count==12);const auto v=pair145::upload_info(**static_cast<Registry*>(registry));std::copy(v.begin(),v.end(),out);
});}
int ds41_registry_add_segmented(void* registry,const ds41_tensor_view*view,const ds41_memory_region*parts,size_t count){return checked([&]{
    need(registry&&view&&view->name&&view->rank<=8&&(view->rank==0||view->shape)&&parts&&count&&count<=100000);
    need(view->encoding>=0&&view->encoding<=int(TensorEncoding::Uniform)&&!view->data);
    need(view->protected_native_rows==0||view->protected_native_rows==1);
    std::vector<int64_t>shape;if(view->rank)shape.assign(view->shape,view->shape+view->rank);
    std::vector<MemoryRegion>segments;segments.reserve(count);
    for(size_t i=0;i<count;i++)segments.push_back({parts[i].data,parts[i].bytes});
    (*static_cast<Registry*>(registry))->add_segmented(view->name,{TensorEncoding(view->encoding),std::move(shape),nullptr,view->bytes,view->bits,bool(view->protected_native_rows)},std::move(segments));
});}
void ds41_registry_destroy(void* registry){delete static_cast<Registry*>(registry);}
void* ds41_checkpoint_create(void*registry,const char*geometry){void*result=nullptr;checked([&]{
    need(registry&&geometry);result=new NativeCheckpoint(*static_cast<Registry*>(registry),geometry);
});return result;}
size_t ds41_checkpoint_dense_bytes(void*c){size_t result=0;checked([&]{need(c);result=static_cast<NativeCheckpoint*>(c)->dense_arena_bytes();});return result;}
int ds41_checkpoint_protect_rows(void*c,const char*t,const char*i,const char*w,const char*s){return checked([&]{
    need(c&&t&&i&&w&&s);static_cast<NativeCheckpoint*>(c)->rows()->protect(t,i,w,s);
});}
void ds41_checkpoint_destroy(void*c){delete static_cast<NativeCheckpoint*>(c);}
void* ds41_rows_create(void*registry){void*result=nullptr;checked([&]{need(registry);result=new BorrowedRows(*static_cast<Registry*>(registry));});return result;}
int ds41_rows_protect(void*rows,const char*t,const char*i,const char*w,const char*s){return checked([&]{
    need(rows&&t&&i&&w&&s);static_cast<BorrowedRows*>(rows)->protect(t,i,w,s);
});}
int ds41_rows_gather(void*rows,const char*name,const int64_t*ids,size_t count,uint16_t*out,size_t size){return checked([&]{
    need(rows&&name&&ids&&out&&count&&count<=1048576&&size&&size<=32ull*1024*1024);
    auto values=static_cast<BorrowedRows*>(rows)->gather_bf16(name,std::vector<int64_t>(ids,ids+count));
    need(values.size()==size);std::copy(values.begin(),values.end(),out);
});}
void ds41_rows_destroy(void*rows){delete static_cast<BorrowedRows*>(rows);}
void* ds41_experts_create(void* registry,const char* tasks,const char* pool,int a8,const int* cpus,size_t count,int coordinator){
    void*result=nullptr;checked([&]{need(registry&&tasks&&pool&&cpus&&count&&count<=256&&(a8==0||a8==1||a8==2));
        result=new NativeExperts(*static_cast<Registry*>(registry),tasks,pool,a8==1,std::vector<int>(cpus,cpus+count),coordinator,a8==2);
    });return result;
}
int ds41_experts_add_layer(void* e,int layer,const char* prefix,int experts,int dim,int inter,int topk,float limit){return checked([&]{
    need(e&&prefix);static_cast<NativeExperts*>(e)->add_layer(layer,prefix,experts,dim,inter,topk,limit);
});}
int ds41_experts_run(void* e,int layer,int rows,const uint16_t* x,size_t nx,const int64_t* ids,size_t ni,const float* routing,size_t nr,uint16_t* out,size_t no){return checked([&]{
    need(e&&x&&ids&&routing&&out&&nx&&ni&&nr&&no&&nx<=65536&&ni<=48&&nr==ni&&no<=393216);
    need(rows>=1&&rows<=8&&nx%size_t(rows)==0&&ni%size_t(rows)==0&&no==nx*(ni/size_t(rows)));
    auto result=static_cast<NativeExperts*>(e)->run(layer,rows,std::vector<uint16_t>(x,x+nx),std::vector<int64_t>(ids,ids+ni),std::vector<float>(routing,routing+nr));
    need(result.size()==no);std::copy(result.begin(),result.end(),out);
});}
int ds41_experts_worker_rows(void* e,uint64_t*out,size_t count){return checked([&]{need(e&&out&&count&&count<=256);
    auto rows=static_cast<NativeExperts*>(e)->worker_rows();need(rows.size()==count);std::copy(rows.begin(),rows.end(),out);
});}
void ds41_experts_destroy(void* e){delete static_cast<NativeExperts*>(e);}
}
