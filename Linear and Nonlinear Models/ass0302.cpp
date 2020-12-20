#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <math.h>
#include <random>
using namespace std;

double dot(double *w, double x[][11], int pickNumber)
{
    double output = 0;
    for (int i = 0; i < 11; i++)
    {
        output = output + w[i] * x[pickNumber][i];
    }
    return output;
}

double theta(double s)
{
    return 1 / (1 + exp(-s));
}
double SqrEin(double *w, double x[][11], double *y)
{
    double error = 0;
    for (int i = 0; i < 1000; i++)
        error = error + (dot(w, x, i) - y[i]) * (dot(w, x, i) - y[i]);
    return error / 1000;
}

int count(double x[][11], double *y, int distrib)
{
    double w[11] = {0};
    double Ein = 1.0;
    const double linSqrEin = 0.60532;
    int count = 0;
    //srand(time(NULL));
    while (Ein > 1.01 * linSqrEin)
    {
        //std::mt19937 gen(rd());
        //std::default_random_engine generator;
        //std::uniform_int_distribution<int> distribution(0, 1000);
        int pickNumber = distrib;
        //cout << pickNumber << endl;
        double sl = 2 * 0.001 * theta(-y[pickNumber] * dot(w, x, pickNumber)) * (y[pickNumber] - dot(w, x, pickNumber));
        for (int i = 0; i < 11; i++)
        {
            w[i] = w[i] + sl * x[pickNumber][i];
        }
        Ein = SqrEin(w, x, y);
        count = count + 1;
    }
    return count;
}

int main()
{
    const int N = 1000;
    fstream file;
    string test;
    double labelY[N];
    double dataX[N][11];
    //vector<vector<double>> dataX[N][11];
    if (!file) //檢查檔案是否成功開啟
    {
        cerr << "Can't open file!\n";
        exit(1); //在不正常情形下，中斷程式的執行
    }
    file.open("hw3_train.txt", ios::in);
    int ccount = 0;
    while (!file.eof())
    {
        getline(file, test);
        string result;
        vector<string> res;
        stringstream input(test);
        while (input >> result)
        {
            res.push_back(result);
        }
        dataX[ccount][0] = 1.0;
        int kount = 1;
        for (int i = 0; i < res.size(); i++)
        {
            if (i == 10)
            {
                labelY[ccount] = stod(res[i]);
            }
            else
            {
                dataX[ccount][kount] = stod(res[i]);
            }
            kount++;
        }
        ccount++;
    }
    std::random_device rd; //Will be used to obtain a seed for the random number engine`
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 1000);
    int accu = 0;
    for (int i = 0; i < 1000; i++)
    {
        double w[11] = {0};
        double Ein = 1.0;
        const double linSqrEin = 0.60532;
        int count = 0;
        //srand(time(NULL));
        while (Ein > 1.01 * linSqrEin)
        {
            int pickNumber = distrib(gen);
            double sl = 2 * 0.001 * theta(-labelY[pickNumber] * dot(w, dataX, pickNumber)) * (labelY[pickNumber] - dot(w, dataX, pickNumber));
            for (int i = 0; i < 11; i++)
            {
                w[i] = w[i] + sl * dataX[pickNumber][i];
            }
            Ein = SqrEin(w, dataX, labelY);
            count = count + 1;
        }
        int tempAns = count;
        accu = accu + tempAns;
        cout << i << " " << tempAns << endl;
    }
    double answer = accu / 1000;
    cout << answer << endl;
    file.close();
    return 0;
}
