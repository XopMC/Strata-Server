// Reuse ONLY the actual linked CPU-v224 Oracle/admission/fixture helpers.
// Its renamed legacy entry point is never called. No copied CUDA math oracle.
#define main v86_reference_entry_not_called
#include "../cpu-semantics-cuda-leaf-v86/component_v86.cpp"
#undef main
#include "../cpu-resident-grouped-chain-v6-r1/grouped_v6.hpp"
namespace G=cpu_grouped_v6;namespace R=cpu_rows_v5;namespace B=four_banks_v3;
namespace {
struct GroupDevice {
 cudaStream_t s;int bank;
 Buffer packed,ids,logical,refs,in,plan,plan_error,physical,x,q,g,u,gs,us,h,hq,d,y,status;
 G::Images images;
 GroupDevice(cudaStream_t stream,int b):s(stream),bank(b),packed(4*B::SlotStride,s),ids(R::Capacity*4,s),logical(B::Keys*4,s),refs(B::Keys*sizeof(B::PhysicalRef),s),in(sizeof(R::Inputs),s),plan(sizeof(R::Plan),s),plan_error(4,s),physical(sizeof(B::PhysicalDeviceView),s),x(V::MaxT*V::H*4,s),q(V::MaxT*V::InputQ,s),g(R::Capacity*V::F*4,s),u(g.n,s),gs(g.n,s),us(g.n,s),h(g.n,s),hq(R::Capacity*V::HiddenQ,s),d(R::Capacity*V::H*4,s),y(d.n,s),status(sizeof(G::Status),s){
  images={plan.as<R::Plan>(),plan_error.as<uint32_t>(),physical.as<B::PhysicalDeviceView>(),x.as<float>(),q.as<uint8_t>(),g.as<float>(),u.as<float>(),gs.as<float>(),us.as<float>(),h.as<float>(),hq.as<uint8_t>(),d.as<float>(),y.as<float>(),status.as<G::Status>()};
 }
 void reset(){for(auto*b:{&q,&g,&u,&gs,&us,&h,&hq,&d,&y,&status})b->reset();}
 void capture(Graph&graph){graph.capture([&]{ck(R::plan_async(in.as<R::Inputs>(),plan.as<R::Plan>(),plan_error.as<uint32_t>(),s),"group route plan");ck(G::chain(images,bank,s),"group five leaves plus publish");});}
 G::Status state(){auto z=status.read();G::Status a{};std::memcpy(&a,z.data(),sizeof a);return a;}
 void guards(){for(auto*b:{&packed,&ids,&logical,&refs,&in,&plan,&plan_error,&physical,&x,&q,&g,&u,&gs,&us,&h,&hq,&d,&y,&status})b->guards();}
};
struct GroupHost {
 int bank;R::Inputs input{};B::PhysicalDeviceView view{};
 std::vector<int32_t>ids,logical;std::vector<B::PhysicalRef>refs;
 std::array<std::vector<unsigned char>,4>weights;
 std::vector<unsigned char>packed,q,hq;
 std::vector<float>x,g,u,gs,us,h,d,y;
 std::array<Oracle,4>oracle;
 R::Plan reference{};unsigned jobs=0,tokens=0;
 GroupHost(int b):bank(b),ids(R::Capacity),logical(B::Keys,-1),refs(B::Keys),packed(4*B::SlotStride),q(V::MaxT*V::InputQ),hq(R::Capacity*V::HiddenQ),x(V::MaxT*V::H),g(R::Capacity*V::F),u(g.size()),gs(g.size()),us(g.size()),h(g.size()),d(R::Capacity*V::H),y(d.size()){for(auto&w:weights)w.resize(V::Blob);}
 void prepare(GroupDevice&a,int T,int profile,int replay){
  input=R::Inputs{};input.T=T;input.layer=(profile*7+replay)%48;input.sequence=1+T*100+profile*2+replay;input.expected_physical_epoch=100+replay;input.expected_logical_epoch=200+replay;
  view=B::PhysicalDeviceView{};view.physical_epoch=input.expected_physical_epoch;
  for(int b=0;b<4;++b){view.bank_bytes[b]=4*B::SlotStride;view.base[b]=b==bank?a.packed.p():reinterpret_cast<const unsigned char*>(uintptr_t(0x100000000ull+uint64_t(b)*0x10000000000ull));}
  // Only this admitted bank's four synthetic original-byte experts are read.
  // Nonlocal refs are symbolic metadata; Root runs this component on each bank.
  std::fill(refs.begin(),refs.end(),B::PhysicalRef{});std::fill(logical.begin(),logical.end(),-1);pad(packed);
  std::vector<float>ignored(x.size());
  for(int slot=0;slot<4;++slot){synthetic(weights[slot],ignored,profile,replay+slot*17+bank*31);
   if(profile==3){const float scales[4][4]={{0.f,.5f,1.f,0.f},{-.75f,.5f,-.3125f,0.f},{0x1p-126f,1.f,1.f,0.f},{8.f,.125f,.125f,0.f}};std::memcpy(weights[slot].data()+V::TailOffset,scales[slot],16);}
   std::memcpy(packed.data()+size_t(slot)*B::SlotStride,weights[slot].data(),V::Blob);}
  std::vector<unsigned char>ignored_weight(V::Blob);synthetic(ignored_weight,x,profile,replay);
  for(int expert=0;expert<16;++expert){int key=input.layer*512+expert;refs[key]={uint64_t(expert/4)*B::SlotStride,input.expected_physical_epoch,expert%4,expert/4,B::BlobBytes,uint32_t(key)};bool hit=profile==4||(profile==1&&(expert/4)%2==0);logical[key]=hit?(expert+3):-1;}
  for(int r=0;r<R::Capacity;++r)ids[r]=profile==3?(r%3)*4+bank:(r/10*3+r%10+replay*5)%16;
  input.ids=ids.data();input.logical={logical.data(),input.expected_logical_epoch};view.refs=refs.data();input.physical=view;
  uint32_t error=0;need(R::reference_plan(input,reference,error)&&!error,"group host route oracle");jobs=0;tokens=0;
  for(auto*v:{&q,&hq})pad(*v);for(auto*v:{&g,&u,&gs,&us,&h,&d,&y})pad(*v);
  std::array<bool,4>needed{};std::array<bool,8>used{};
  for(int r=0;r<T*10;++r)if(reference.jobs[r].kind==R::Cpu&&reference.jobs[r].physical.device==bank){++jobs;used[r/10]=true;needed[reference.jobs[r].physical.physical_slot]=true;}
  for(int slot=0;slot<4;++slot)if(needed[slot])oracle[slot].run(weights[slot],x,T);
  for(int t=0;t<T;++t)if(used[t]){++tokens;int slot=0;while(!needed[slot])++slot;std::memcpy(q.data()+size_t(t)*V::InputQ,oracle[slot].q.data()+size_t(t)*V::InputQ,V::InputQ);}
  for(int r=0;r<T*10;++r){const auto&job=reference.jobs[r];if(job.kind!=R::Cpu||job.physical.device!=bank)continue;const auto&o=oracle[job.physical.physical_slot];const int t=job.token;
   std::memcpy(hq.data()+size_t(r)*V::HiddenQ,o.hq.data()+size_t(t)*V::HiddenQ,V::HiddenQ);
   auto copy=[&](std::vector<float>&dst,const std::vector<float>&src,int n){std::memcpy(dst.data()+size_t(r)*n,src.data()+size_t(t)*n,n*4);};
   copy(g,o.g,V::F);copy(u,o.u,V::F);copy(gs,o.gs,V::F);copy(us,o.us,V::F);copy(h,o.h,V::F);copy(d,o.d,V::H);copy(y,o.y,V::H);
  }
 }
 void upload(GroupDevice&a){
  a.packed.put(packed.data(),packed.size());a.ids.put(ids.data(),ids.size()*4);a.logical.put(logical.data(),logical.size()*4);a.refs.put(refs.data(),refs.size()*sizeof refs[0]);a.x.put(x.data(),x.size()*4);
  R::Inputs dev=input;dev.ids=a.ids.as<int32_t>();dev.logical.virtual_slots=a.logical.as<int32_t>();dev.physical.refs=a.refs.as<B::PhysicalRef>();a.in.put(&dev,sizeof dev);
  B::PhysicalDeviceView dv=view;dv.refs=a.refs.as<B::PhysicalRef>();a.physical.put(&dv,sizeof dv);a.reset();
 }
 void compare(GroupDevice&a){
  auto st=a.state();need(st.error==0,"group producer error");need(st.input_tokens==tokens&&st.jobs==jobs&&st.output_rows==jobs,"group counters not exact");need(st.stamp.sequence==input.sequence&&st.stamp.logical_epoch==input.expected_logical_epoch&&st.stamp.physical_epoch==input.expected_physical_epoch,"group stamp mismatch");
  a.q.equal(q.data(),q.size(),"shared original input Q8");a.hq.equal(hq.data(),hq.size(),"group hidden Q8");
  a.g.equal(g.data(),g.size()*4,"group raw gate");a.u.equal(u.data(),u.size()*4,"group raw up");a.gs.equal(gs.data(),gs.size()*4,"group scaled gate");a.us.equal(us.data(),us.size()*4,"group scaled up");a.h.equal(h.data(),h.size()*4,"group hidden");a.d.equal(d.data(),d.size()*4,"group raw down");a.y.equal(y.data(),y.size()*4,"group CPU row output");a.packed.equal(packed.data(),packed.size(),"group original packed bytes immutable");a.x.equal(x.data(),x.size()*4,"group inputs immutable");a.guards();
 }
};
void group_refusals(GroupDevice&a,GroupHost&h,Graph&graph){
 for(int test=0;test<4;++test){h.prepare(a,1,0,0);h.upload(a);
  if(test==0){auto x=h.x;x[0]=INFINITY;a.x.put(x.data(),x.size()*4);}
  if(test==1){auto packed=h.packed;uint32_t nan=0x7fc12345u;std::memcpy(packed.data()+V::TailOffset,&nan,4);a.packed.put(packed.data(),packed.size());}
  if(test==2){auto v=h.view;v.refs=a.refs.as<B::PhysicalRef>();++v.physical_epoch;a.physical.put(&v,sizeof v);}
  if(test==3){auto ids=h.ids;ids[0]=-1;a.ids.put(ids.data(),ids.size()*4);}
  graph.launch();auto st=a.state();need(st.error!=0,"group refusal missing");need(st.stamp.sequence==0&&st.stamp.logical_epoch==0&&st.stamp.physical_epoch==0,"failed group published success stamp");a.guards();
 }
 auto bad=a.images;bad.output=bad.down_raw;need(G::chain(bad,a.bank,a.s)==cudaErrorInvalidValue,"group alias admitted");need(G::chain(a.images,4,a.s)==cudaErrorInvalidValue,"group GPU4 admitted");
 std::printf("V6_GROUP_REFUSAL domain_or_plan=4 abi=2 stamp_zero=1\n");
}
}
