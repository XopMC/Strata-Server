    struct Bulk418 {
        using Describe=int(*)(const void*,Ds143PairDescriptor*);
        using Info=int(*)(void*,uint64_t*,int);using Cpus=int(*)(void*,int*,int);
        struct Job {
            Task*task;const std::vector<int>*selected;const std::vector<float>*alpha;
            std::array<std::vector<int>,2>slots;
            std::array<uint64_t,3>calls[2]{};
            std::atomic<bool>cancelled{false};int status[2]{};uint64_t helper_spins427=0,helper_parks427=0;
        };
        struct alignas(64) Mailbox {std::mutex mutex;std::condition_variable cv;Job*job=nullptr;uint64_t published=0,completed=0;bool stop=false,ready=false;int affinity=0;};
        static constexpr uint64_t ActiveNs427=10000000;
        struct alignas(64) Published427 {std::atomic<uint64_t>epoch{0};std::atomic<Job*>job{nullptr};};
        struct alignas(64) Completed427 {std::atomic<uint64_t>epoch{0};};
        struct alignas(64) ActiveMailbox427 {
            Published427 published;Completed427 completed;
            std::mutex park_mutex;std::condition_variable cv;
            std::atomic<bool>stop{false},helper_parked{false},main_parked{false};
        };
        static_assert(offsetof(ActiveMailbox427,completed)>=64,"427 epochs must not false-share");
        bool active427=false;
        ds418_memory::HomeArray<ActiveMailbox427>active_mail427;
        Engine&engine;bool component419=false;int poison=0;void*home[2]{};int count[2]{};std::vector<int>cpus[2];
        Describe describe=nullptr;Info info=nullptr;
        ds418_memory::HomeArray<uint64_t>done0,done1;
        ds418_memory::HomeArray<int>actual0,actual1;
        ds418_memory::HomeArray<Mailbox>mail;
        std::thread helper;std::mutex command;
        std::shared_ptr<Receipt418>receipt=std::make_shared<Receipt418>();
        Bulk418(Engine&e,const Ds418Pools&b,void*library):active_mail427(1,0),engine(e),done0(b.workers[0],0),done1(b.workers[1],1),actual0(b.workers[0],0),actual1(b.workers[1],1),mail(1,0){
            const char*poll=std::getenv("DS1017_CPU_NUMA_BULK_ACTIVE_EPOCH_V427");if(poll&&std::strcmp(poll,"0")&&std::strcmp(poll,"1"))throw std::runtime_error("427 literal active-epoch flag required");active427=poll&&!std::strcmp(poll,"1");
            const char*test=std::getenv("DS1017_CPU_NUMA_BULK_COMPONENT_V419");if(test&&std::strcmp(test,"0")&&std::strcmp(test,"1"))throw std::runtime_error("418 literal component flag required");component419=test&&!std::strcmp(test,"1");
            describe=reinterpret_cast<Describe>(dlsym(library,"ds418_paired_descriptor"));
            info=reinterpret_cast<Info>(dlsym(library,"ds418_home_pool_info"));
            auto ids=reinterpret_cast<Cpus>(dlsym(library,"ds418_home_pool_cpus"));
            if(!describe||!info||!ids)throw std::runtime_error("418 actual CPU home APIs required");
            for(int h=0;h<2;h++){
                home[h]=b.pool[h];count[h]=b.workers[h];std::array<uint64_t,8>v{};
                if(info(home[h],v.data(),8)||v[0]!=418||v[1]!=uint64_t(h+1)||v[2]!=uint64_t(count[h])||v[3+h]!=v[2]||v[4-h])throw std::runtime_error("418 genuine home pool required");
                cpus[h].resize(count[h]);if(ids(home[h],cpus[h].data(),count[h]))throw std::runtime_error("418 actual CPU list required");
                for(size_t i=0;i<cpus[h].size();i++)if(cpus[h][i]<0||cpus[h][i]>253||cpus[h][i]/128!=h||cpus[h][i]==127||(i&&cpus[h][i]<=cpus[h][i-1]))throw std::runtime_error("418 home CPUs differ");
            }
            {std::lock_guard<std::mutex>l(receipt_registry418_mutex);if(!receipt_registry418.expired())throw std::runtime_error("418 single enabled Taskengine required");receipt_registry418=receipt;}
            receipt->v={418,1,uint64_t(count[0]),uint64_t(count[1]),127,254,0,0,0,0,0,0,0,0,0,uint64_t(component419)};
            receipt->epoch427={427,uint64_t(active427),ActiveNs427,0,0,0,0,0,0,0,0,0,0,0,127,254};
            helper=std::thread([this]{
                cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(127,&mask);int affinity=sched_setaffinity(0,sizeof(mask),&mask)||sched_getcpu()!=127?6:0;
                auto&m=mail[0];std::unique_lock<std::mutex>l(m.mutex);m.affinity=affinity;m.ready=true;m.cv.notify_all();
                if(active427){l.unlock();if(!affinity)active_helper427(affinity);return;}
                for(;;){m.cv.wait(l,[&]{return m.stop||m.published!=m.completed;});if(m.stop)return;Job*j=m.job;uint64_t sequence=m.published;l.unlock();
                    if(affinity){j->status[0]=affinity;j->cancelled.store(true,std::memory_order_release);}else domain(0,*j);
                    l.lock();m.completed=sequence;m.job=nullptr;m.cv.notify_all();
                }
            });
            auto&m=mail[0];std::unique_lock<std::mutex>l(m.mutex);m.cv.wait(l,[&]{return m.ready;});if(m.affinity){m.stop=true;m.cv.notify_all();l.unlock();helper.join();throw std::runtime_error("418 helper127 affinity refused");}
        }
        ~Bulk418(){
            if(active427){auto&m=active_mail427[0];uint64_t spins=0,parks=0;wait_done427(m.published.epoch.load(std::memory_order_acquire),spins,parks);
                {std::lock_guard<std::mutex>l(m.park_mutex);m.stop.store(true,std::memory_order_release);m.cv.notify_all();}
            }else{auto&m=mail[0];std::unique_lock<std::mutex>l(m.mutex);m.cv.wait(l,[&]{return m.published==m.completed;});m.stop=true;m.cv.notify_all();}
            if(helper.joinable())helper.join();
        }
        static uint64_t clock427(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
        void active_helper427(int affinity)noexcept{
            auto&m=active_mail427[0];uint64_t observed=0,spins=0,parks=0,start=clock427();unsigned polls=0;
            for(;;){
                if(m.stop.load(std::memory_order_acquire))return;
                uint64_t next=m.published.epoch.load(std::memory_order_acquire);
                if(next==observed){
                    ++spins;__builtin_ia32_pause();
                    if(++polls%256==0&&clock427()-start>=ActiveNs427){
                        std::unique_lock<std::mutex>l(m.park_mutex);
                        m.helper_parked.store(true,std::memory_order_release);
                        // Predicate recheck occurs under the same park mutex
                        // used by a publisher that observes helper_parked.
                        if(!m.stop.load(std::memory_order_acquire)&&m.published.epoch.load(std::memory_order_acquire)==observed){++parks;m.cv.wait(l,[&]{return m.stop.load(std::memory_order_acquire)||m.published.epoch.load(std::memory_order_acquire)!=observed;});}
                        m.helper_parked.store(false,std::memory_order_release);start=clock427();polls=0;
                    }continue;
                }
                Job*j=m.published.job.load(std::memory_order_acquire);if(!j)std::terminate();
                if(affinity){j->status[0]=affinity;j->cancelled.store(true,std::memory_order_release);}else domain(0,*j);
                j->helper_spins427=spins;j->helper_parks427=parks;observed=next;
                // Clear borrowed pointer BEFORE publishing completion. A caller
                // that acquires completion can then safely destroy/reuse Job.
                {std::lock_guard<std::mutex>l(m.park_mutex);
                    m.published.job.store(nullptr,std::memory_order_release);
                    m.completed.epoch.store(next,std::memory_order_release);
                    if(m.main_parked.load(std::memory_order_acquire))m.cv.notify_all();
                }
                spins=parks=0;start=clock427();polls=0;
            }
        }
        void wait_done427(uint64_t wanted,uint64_t&spins,uint64_t&parks){
            auto&m=active_mail427[0];uint64_t start=clock427();unsigned polls=0;
            while(m.completed.epoch.load(std::memory_order_acquire)!=wanted){
                ++spins;__builtin_ia32_pause();
                if(++polls%256==0&&clock427()-start>=ActiveNs427){
                    std::unique_lock<std::mutex>l(m.park_mutex);m.main_parked.store(true,std::memory_order_release);
                    if(m.completed.epoch.load(std::memory_order_acquire)!=wanted){++parks;m.cv.wait(l,[&]{return m.completed.epoch.load(std::memory_order_acquire)==wanted;});}
                    m.main_parked.store(false,std::memory_order_release);start=clock427();polls=0;
                }
            }
        }
        int validate_task(Task&t){
            if(t.experts!=(component419?18:384)||t.dim!=5120||t.inter!=2304||t.topk!=6||t.rows<1||t.rows>6||t.limit!=10.f)return 1;
            t.home418.resize(t.experts);uint64_t generation=0,epoch=0;
            for(int e=0;e<t.experts;e++)for(int r=0;r<3;r++){
                Ds143PairDescriptor d{};int i=e*3+r;if(describe(reinterpret_cast<const void*>(t.weights[i]),&d)||d.struct_bytes!=sizeof(d)||d.layout_tag!=139||d.bits||d.bytes!=6635520||d.expert!=e||d.role!=r||d.rows!=(r==2?t.dim:t.inter)||d.columns!=(r==2?t.inter:t.dim)||t.formats[i]||t.scales[i]!=t.weights[i]+512||!d.owner_generation||!d.bank_format_epoch)return 1;
                if(t.layer418<0)t.layer418=d.layer;if(d.layer!=t.layer418)return 1;
                if(!generation){generation=d.owner_generation;epoch=d.bank_format_epoch;}if(generation!=d.owner_generation||epoch!=d.bank_format_epoch)return 1;
                t.home418[e]=(d.layer+e)%2;
            }return 0;
        }
        struct MappedPost {PostContext original;const std::vector<int>*slots;};
        static int mapped_post(void*raw,int begin,int end){
            auto&c=*static_cast<MappedPost*>(raw);constexpr int blocks=72;
            while(begin<end){int local=begin/blocks,offset=begin%blocks,n=std::min(end-begin,blocks-offset);int original=(*c.slots)[local]*blocks+offset;
                int rc=Engine::hidden_post(&c.original,original,original+n);if(rc)return rc;begin+=n;
            }return 0;
        }
        int project(int h,Job&j,bool first){
            auto&t=*j.task;uintptr_t w[96]{},s[96]{},x[96]{},out[96]{};int n[96]{},k[96]{},bits[96]{};Ds41A8ViewV118 views[96]{};int entries=0;uint64_t expected=0;
            for(int slot:j.slots[h])if((*j.alpha)[slot]!=0.f)for(int role=first?0:2;role<(first?2:3);role++){
                if(entries>=96)return 1;int index=(*j.selected)[slot]*3+role;
                w[entries]=t.weights[index];s[entries]=t.scales[index];n[entries]=first?t.inter:t.dim;k[entries]=first?t.dim:t.inter;
                x[entries]=uintptr_t(first?t.activation.data()+(slot/t.topk)*t.dim:t.hidden.data()+slot*t.inter);out[entries]=uintptr_t(first?t.gate.data()+(slot*2+role)*t.inter:t.down.data()+slot*t.dim);
                if(first){int q=slot/t.topk;views[entries]={t.gu_codes.data()+q*t.dim,t.gu_scales.data()+q*(t.dim/32),t.gu_sums.data()+q*(t.dim/4),t.dim};}
                else views[entries]={t.hidden_codes.data()+slot*t.inter,t.hidden_scales.data()+slot*(t.inter/32),t.hidden_sums.data()+slot*(t.inter/4),t.inter};
                expected+=n[entries];entries++;
            }
            if(!entries)return 0;auto*d=h?done1.data():done0.data();auto*a=h?actual1.data():actual0.data();
            int rc=engine.prequant_batch(home[h],w,s,x,out,n,k,bits,entries,views,d,a);if(rc)return rc;
            uint64_t rows=0;for(int i=0;i<count[h];i++){if(a[i]!=cpus[h][i])return 6;rows+=d[i];engine.total_work[h?count[0]+i:i].fetch_add(d[i],std::memory_order_relaxed);}if(rows!=expected)return 9;
            j.calls[h][first?0:2]++;return 0;
        }
        void domain(int h,Job&j)noexcept{
            try{
                if(j.cancelled.load(std::memory_order_acquire))return;
                int rc=project(h,j,true);if(rc){j.status[h]=rc;j.cancelled.store(true,std::memory_order_release);return;}
                if(j.cancelled.load(std::memory_order_acquire))return;
                if(j.calls[h][0]){MappedPost pc{{j.task,j.alpha,engine.quant_exact},&j.slots[h]};rc=engine.post(home[h],mapped_post,&pc,int(j.slots[h].size())*72);if(rc){j.status[h]=rc;j.cancelled.store(true,std::memory_order_release);return;}j.calls[h][1]++;}
                if(j.cancelled.load(std::memory_order_acquire))return;
                rc=project(h,j,false);if(rc){j.status[h]=rc;j.cancelled.store(true,std::memory_order_release);}
            }catch(...){j.status[h]=9;j.cancelled.store(true,std::memory_order_release);}
        }
        int run(Task&t,const std::vector<int>&selected,const std::vector<float>&alpha){
            std::unique_lock<std::mutex>serial(command,std::try_to_lock);if(!serial.owns_lock()||sched_getcpu()!=254)return 6;
            if(poison)return poison;
            if(t.home418.size()!=size_t(t.experts)||selected.size()!=size_t(t.rows*t.topk)||alpha.size()!=selected.size())return 1;
            Job j{&t,&selected,&alpha};for(size_t i=0;i<selected.size();i++){if(selected[i]<0||selected[i]>=t.experts||!std::isfinite(alpha[i])||alpha[i]<0)return 1;j.slots[t.home418[selected[i]]].push_back(int(i));}
            // Query FP8 and the original118 A8 view are completely produced by
            // coordinator254 before this release. No home rewrites shared input.
            {std::lock_guard<std::mutex>l(receipt->mutex);receipt->v[14]=1;}
            uint64_t sequence=0,main_spins=0,main_parks=0;
            if(active427){
                auto&m=active_mail427[0];sequence=m.published.epoch.load(std::memory_order_acquire);
                if(m.stop.load(std::memory_order_acquire)||m.completed.epoch.load(std::memory_order_acquire)!=sequence||sequence==UINT64_MAX){std::lock_guard<std::mutex>r(receipt->mutex);receipt->v[14]=0;return 9;}
                // The guard outlives publication and ALWAYS drains a borrowed
                // stack Job on exceptional unwind. No throwing operation lies
                // between epoch publication and arming this guard.
                struct Drain427 {Bulk418*self;uint64_t epoch=0;bool armed=false;
                    ~Drain427()noexcept{if(armed){try{uint64_t s=0,p=0;self->wait_done427(epoch,s,p);}catch(...){std::terminate();}}}
                }drain{this};
                ++sequence;drain.epoch=sequence;
                {std::lock_guard<std::mutex>l(m.park_mutex);
                    m.published.job.store(&j,std::memory_order_release);
                    m.published.epoch.store(sequence,std::memory_order_release);drain.armed=true;
                    if(m.helper_parked.load(std::memory_order_acquire))m.cv.notify_all();
                }
                domain(1,j);wait_done427(sequence,main_spins,main_parks);drain.armed=false;
            }else{
            auto&m=mail[0];uint64_t sequence;
            {std::lock_guard<std::mutex>l(m.mutex);if(m.stop||m.published!=m.completed){std::lock_guard<std::mutex>r(receipt->mutex);receipt->v[14]=0;return 9;}m.job=&j;sequence=++m.published;m.cv.notify_all();}
            domain(1,j);
            {std::unique_lock<std::mutex>l(m.mutex);m.cv.wait(l,[&]{return m.completed==sequence;});}
            }
            int rc=j.status[0]?j.status[0]:j.status[1];if(j.cancelled.load(std::memory_order_acquire)&&!rc)rc=9;
            if(rc)poison=rc;
            {std::lock_guard<std::mutex>l(receipt->mutex);auto&v=receipt->v;if(rc)v[13]++;else v[6]++;for(int h=0;h<2;h++)for(int phase=0;phase<3;phase++)v[7+h*3+phase]+=j.calls[h][phase];
                if(active427){auto&a=receipt->epoch427;a[3]++;a[4]++;a[5]++;a[6]+=j.helper_spins427;a[7]+=main_spins;a[8]+=j.helper_parks427;a[9]+=main_parks;a[10]=sequence;a[11]=active_mail427[0].completed.epoch.load(std::memory_order_acquire);if(rc)a[12]++;}
                // Both receipts become idle atomically under their ONE mutex.
                v[14]=0;
            }
            return rc;
        }
    };
    void enable_bulk418(const Ds418Pools&b){
        const char*f=std::getenv("DS1017_CPU_NUMA_BULK_V418");if(!f||std::strcmp(f,"1")||!a8||coordinator!=254)throw std::runtime_error("418 explicit production mode required");
        Dl_info where{};if(!dladdr(reinterpret_cast<void*>(b.gemv_function),&where)||!where.dli_fname)throw std::runtime_error("418 CPU library identity");
        void*lib=dlopen(where.dli_fname,RTLD_NOW|RTLD_NOLOAD);if(!lib)throw std::runtime_error("418 CPU library missing");
        try{bulk418=std::make_unique<Bulk418>(*this,b,lib);}catch(...){dlclose(lib);throw;}dlclose(lib);
    }
    int bulk_all418(Task&t,const std::vector<int>&selected,const std::vector<float>&alpha){
        std::array<bool,6>active{};for(size_t i=0;i<alpha.size();i++)if(alpha[i]!=0.f)active[i/t.topk]=true;
        for(int q=0;q<t.rows;q++)if(active[q]){int rc=quant_exact(t.activation.data()+q*t.dim,t.dim,t.gu_codes.data()+q*t.dim,t.gu_scales.data()+q*(t.dim/32),t.gu_sums.data()+q*(t.dim/4));if(rc)return rc;}
        for(size_t i=0;i<alpha.size();i++)if(alpha[i]==0.f)std::fill_n(t.down.data()+i*t.dim,t.dim,0.f);
        return bulk418->run(t,selected,alpha);
    }
