#include "NN.h"

using namespace std;
using namespace tensorflow;
using namespace ops;
using namespace chrono;

namespace nn{
	Tensor P2Tensor(position c_position){
		Tensor input(DT_FLOAT, TensorShape({2, 19, 19, 2}));
		auto input_map = input.tensor<float, 4>();
		vector<vector<int>> board = c_position.getboard();
		int player;
		for(int i = 0;i<19;++i){
			for(int j = 0;j<19;++j){
				player = board[i][j];
				input_map(0, i, j, 0) = player==-1;
				input_map(1, i, j, 0) = player==1;
				input_map(0, i, j, 1) = player==1;
				input_map(1, i, j, 1) = player==-1;
			}
		}
		return input;
	}
}
