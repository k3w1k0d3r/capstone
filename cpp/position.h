#ifndef POSITION_H
#define POSITION_H
#include <iostream>
#include <vector>

using namespace std;

class position{
	vector<vector<int>> board;
	int move;
	int movecount;
	public:
		position();
		position(vector<vector<int>> board, int move, int movecount);
		vector<vector<int>> getboard();
		void setstate(vector<vector<int>> board, int move, int movecount);
		int getmove();
		int getmovecount();
		vector<int> getboardflat();
		vector<long> getdims();
};

#endif
