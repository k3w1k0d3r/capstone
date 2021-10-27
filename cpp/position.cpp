#include "position.h"

using namespace std;

position::position(){
	vector<int> row;
	for(int i = 0;i<19;++i){
		row.clear();
		for(int j = 0;j<19;++j){
			row.push_back(0);
		}
		this->board.push_back(row);
	}
	this->move = -1;
	this->movecount = 0;
}
position::position(vector<vector<int>> board, int move, int movecount){
	this->setstate(board, move, movecount);
}
vector<vector<int>> position::getboard(){
	return this->board;
}
void position::setstate(vector<vector<int>> board, int move, int movecount){
	this->board = board;
	this->move = move;
	this->movecount = movecount;
}
int position::getmove(){
	return this->move;
}
vector<int> position::getboardflat(){
	vector<int> boardflat(361, 0);
	for(int i = 0;i<19;++i){
		copy(this->board[i].begin(), this->board[i].end(), boardflat.begin()+19*i);
	}
	return boardflat;
}
int position::getmovecount(){
	return this->movecount;
}
vector<long> position::getdims(){
	return {0, 19, 19};
}
