#define main netflix_existing_cli_main
#include "../src/netflix.cpp"
#undef main

int main(int argc,char** argv){
    try{
        if(argc!=6)fail("expected dataset model movie-output projection user-output");
        auto d=load_dataset(argv[1]);
        auto t=build_temporal_index(d);
        auto m=load_joint(argv[2],d,t,joint_fingerprint(d));
        std::ofstream movie(argv[3],std::ios::binary);
        write_vector(movie,m.factors.item_factors);
        if(std::string(argv[4])=="dump")return 0;
        std::ifstream in(argv[4],std::ios::binary);
        std::vector<float> axes,effective,output(size_t(d.header.users)*12);
        read_vector(in,axes,size_t(m.rank)*12);
        build_effective_users(d,m.factors,m.rank,effective);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,d.header.users,12,m.rank,1.f,
            effective.data(),m.rank,axes.data(),12,0.f,output.data(),12);
        std::ofstream file(argv[5],std::ios::binary);write_vector(file,output);
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
