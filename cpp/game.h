#ifndef GAME_H
#define GAME_H
#include <iostream>
#include <vector>
#include <tuple>
#include "custom_position.h"

using namespace std;

class game{
	vector<int> moveset;
	position i_position = position();
	public:
		game();
		int getplayer(int turn);
		position getposition(int move, position p_position, int player);
		position getposition();
		vector<int> getmoveset();
		void setmoveset(vector<int> moveset);
		bool movelegal(position c_position, int move);
		tuple<bool, int> outcome(position c_position);
		vector<int> getplayerorder();
};

#endif
