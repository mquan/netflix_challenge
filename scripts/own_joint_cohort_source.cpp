#define main netflix_existing_cli_main
#include "../src/netflix.cpp"
#undef main

int main(int argc,char** argv) {
    try {
        if(argc<5 || (argc-3)%2!=0)fail("expected dataset, unlabeled queries and model/output pairs");
        auto d=load_dataset(argv[1]);
        const auto fingerprint=joint_fingerprint(d);
        const auto temporal=build_temporal_index(d);
        std::ifstream extra(argv[2],std::ios::binary|std::ios::ate);
        if(!extra || extra.tellg()!=std::streamoff(2817131*sizeof(ProbeRec)))fail("unlabeled query size");
        extra.seekg(0);
        std::vector<ProbeRec> q;
        read_vector(extra,q,2817131);
        for(const auto& x:q)if(x.rating || x.user>=d.header.users || !x.item || x.item>d.header.items)fail("invalid query address");
        // The fingerprint above binds the training and implicit histories to the saved model.
        for(auto& x:d.probe)x.rating=0;
        d.probe.insert(d.probe.end(),q.begin(),q.end());
        d.header.probe_count=d.probe.size();
        const auto f=joint_features(d,temporal);
        for(int j=3;j<argc;j+=2) {
            auto model=load_joint(argv[j],d,temporal,fingerprint);
            std::vector<float> prediction;
            eval_joint(d,f,model,prediction);
            std::ofstream out(argv[j+1],std::ios::binary);
            write_vector(out,prediction);
            std::cout<<argv[j]<<" rows "<<prediction.size()<<std::endl;
        }
    } catch(const std::exception& ex){std::cerr<<ex.what()<<std::endl;return 1;}
}
