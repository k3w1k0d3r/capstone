#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "search.h"
#include <Python.h>
#include <numpy/arrayobject.h>

using namespace search;
using namespace std;

static PyObject *playgame_exten(PyObject *self, PyObject *args, PyObject *keywds){
	bool use_queue;
	float temperature;
	char *path;
	int thread_count;
	int iterations;
	float epsilon;
	float Alpha;
	static char *kwlist[] = {(char*)"use_queue", (char*)"temperature", (char*)"path", (char*)"iterations", (char*)"thread_count", (char*)"epsilon", (char*)"Alpha", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, keywds, "bfsiiff", kwlist, &use_queue, &temperature, &path, &iterations, &thread_count, &epsilon, &Alpha)){
		return NULL;
	}
	unique_ptr<Session> session;
	nn::LoadGraph(path, &session);
	game p_game = game();
	tuple<int, vector<int>, vector<double>, vector<long>, vector<long>> result = playgame(p_game, &session, temperature, use_queue, iterations, thread_count, epsilon, Alpha);
	int winner = get<0>(result);
	vector<int> boardlist = get<1>(result);
	vector<double> policylist = get<2>(result);
	npy_intp *dims = get<3>(result).data();
	npy_intp *p_dims = get<4>(result).data();
	PyArrayObject *history = (PyArrayObject*)PyArray_SimpleNew(get<3>(result).size(), dims, NPY_INT); //simplenewfromdata had some corruption when returning to python, I suspect useafterfree
	PyArrayObject *p_history = (PyArrayObject*)PyArray_SimpleNew(get<4>(result).size(), p_dims, NPY_DOUBLE);
	memcpy(PyArray_DATA(history), boardlist.data(), boardlist.size()*sizeof(int));
	memcpy(PyArray_DATA(p_history), policylist.data(), policylist.size()*sizeof(double));
	return Py_BuildValue("OOi", PyArray_Return(history), PyArray_Return(p_history), winner);
}
static PyObject *testgame_exten(PyObject *self, PyObject *args, PyObject *keywds){
	srand(time(NULL));
	bool use_queue;
	float temperature;
	char *path1;
	char *path2;
	int thread_count;
	int iterations;
	float epsilon;
	float Alpha;
	static char *kwlist[] = {(char*)"use_queue", (char*)"temperature", (char*)"path1", (char*)"path2", (char*)"iterations", (char*)"thread_count", (char*)"epsilon", (char*)"Alpha", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, keywds, "bfssiiff", kwlist, &use_queue, &temperature, &path1, &path2, &iterations, &thread_count, &epsilon, &Alpha)){
		return NULL;
	}
	unique_ptr<Session> session1;
	unique_ptr<Session> session2;
	bool tested;
	if(rand()%2){
		nn::LoadGraph(path1, &session1);
		nn::LoadGraph(path2, &session2);
		tested = 1;
	}
	else{
		nn::LoadGraph(path1, &session2);
		nn::LoadGraph(path2, &session1);
		tested = 0;
	}
	game p_game = game();
	bool result = testgame(p_game, &session1, &session2, temperature, use_queue, iterations, thread_count, epsilon, Alpha, tested);
	return PyLong_FromLong((long)result);
}
static PyMethodDef MCTS_Methods[] = {
	{"playgame", (PyCFunction)(void(*)(void))playgame_exten, METH_VARARGS|METH_KEYWORDS, "Play a game"},
	{"testgame", (PyCFunction)(void(*)(void))testgame_exten, METH_VARARGS|METH_KEYWORDS, "Test a game"},
	{NULL, NULL, 0, NULL}
};
static struct PyModuleDef MCTS_exten = {
	PyModuleDef_HEAD_INIT,
	"MCTS_exten",
	NULL,
	-1,
	MCTS_Methods
};
PyMODINIT_FUNC PyInit_MCTS_exten(void){
	import_array();
	return PyModule_Create(&MCTS_exten);
}
