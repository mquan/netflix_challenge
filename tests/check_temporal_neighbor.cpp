#define main netflix_cli_main
#include "../src/netflix.cpp"
#undef main
int main() {
  if (double_after({}, "--tiny", 1e-7) != 1e-7) fail("tiny default rounded away");
  Dataset d; d.header.users=1; d.header.items=2;
  d.train_offsets={0,1}; d.train_items={2}; d.train_days={10};
  d.implicit_offsets={0,1}; d.implicit_items={2}; d.implicit_days={10};
  NeighborGraph graph; graph.items=2;graph.k=1;graph.neighbor={1,0};graph.similarity={1,1};
  std::vector<float> residual={2};
  IntegratedNeighbors n(d,graph,residual);n.temporal=true;n.w[0]=.3;n.c[0]=.1;n.beta[0]=.01;
  n.fill_user(d,0);n.correction(0,0,30);float analytic=n.decay_gradient;
  n.beta[0]+=.00001;float plus=n.correction(0,0,30);
  n.beta[0]-=.00002;float minus=n.correction(0,0,30);
  float numeric=(plus-minus)/.00002;
  if(std::abs(analytic-numeric)>.01) fail("temporal derivative mismatch");
  n.beta[0]=0;n.correction(0,0,30);n.update(-1,0,0,1e-7,.01);
  if(n.beta[0]<=0) fail("decay failed to learn");
  std::cout << "temporal gradient and learning verified\n";
}
