#include <torch/script.h> // One-stop header.
// #include <torch/csrc/api/include/torch/serialize.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";


  std::vector<torch::jit::IValue> input;

  torch::Tensor state_goal = torch::ones({1,8}).to(at::kCUDA);
  input.push_back(state_goal);
  auto env_vox = torch::zeros({1, 1, 32, 32});
  // torch::load(env_vox, "/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/cpp/output/env_vox_3.pt");

  input.push_back(env_vox.to(at::kCUDA));

  state_goal[0][0]=1.57;


  at::Tensor output = module.forward(input).toTensor();
  for(size_t i = 0; i < 4; i++){
    std::cout << (output[0][i].item<float>())<< std::endl;
  }

  std::cout << input.at(0) <<std::endl;

}