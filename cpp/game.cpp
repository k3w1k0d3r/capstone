#include "game.h"

using namespace std;

game::game(){
	vector<int> moveset;
	for(int i = 0;i<361;++i){
		moveset.push_back(i);
	}
	this->setmoveset(moveset);
}
int game::getplayer(int turn){
	turn-=1;
	return ((turn%4==0)||(turn%4==1))*2-1;
}
vector<int> game::getplayerorder(){
	return {-1, 1};
}
position game::getposition(int move, position p_position, int player){
	vector<vector<int>> c_board = p_position.getboard();
	c_board[move/19][move%19] = player;
	return position(c_board, move, p_position.getmovecount()+1);
}
position game::getposition(){
	return this->i_position;
}
vector<int> game::getmoveset(){
	return this->moveset;
}
void game::setmoveset(vector<int> moveset){
	this->moveset = moveset;
}
bool game::movelegal(position c_position, int move){
	return c_position.getboard()[move/19][move%19]==0;
}
tuple<bool, int> game::outcome(position c_position){
	vector<vector<int>> board = c_position.getboard();
	int move = c_position.getmove();
	if(move==-1){
		return make_tuple(false, 0);
	}
	int row = move/19;
	int column = move%19;
	int player = board[row][column];
	if(player==0){
		return make_tuple(false, 0);
	}
	int lower;
	int upper;
	int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
	for(int j = 0;j<4;++j){
		lower = -1;
		upper = -1;
		for(int i = 1;i<6;++i){
			if(lower==-1){
				if(row-i*dirs[j][0]<0||column-i*dirs[j][1]<0){
					lower = i;
				}
				else if(board[row-i*dirs[j][0]][column-i*dirs[j][1]]!=player){
					lower = i;
				}
			}
			if(upper==-1){
				if(row+i*dirs[j][0]>18||column+i*dirs[j][1]>18){
					upper = i;
				}
				else if(board[row+i*dirs[j][0]][column+i*dirs[j][1]]!=player){
					upper = i;
				}
			}
			if(lower!=-1&&upper!=-1){
				break;
			}
		}
		if(lower==-1||upper==-1||upper+lower>6){
			return make_tuple(true, player);
		}
	}
	if(c_position.getmovecount()>360){
		return make_tuple(true, 0);
	}
	return make_tuple(false, 0);
}
