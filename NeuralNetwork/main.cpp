//
//  main.cpp
//  NeuralNetwork
//
//  Created by Omar Abdul Rahman on 7/17/16.
//  Copyright © 2016 Omar Abdul Rahman. All rights reserved.
//

#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;


double lambda = 1;
int input_layer_size  = 400;  // 20x20 Input Images of Digits
int hidden_layer_size = 25;   // 25 hidden units
int num_labels = 10;          // 10 labels, from 0 to 9



class NN {
    Col<int> Layers;
    field<mat> Theta;
    
public:
    NN(Col<int> layers);
    static mat sigmoid(mat z);
    static mat sigmoidGradient(mat z);
    static mat randInitializeWeights(int L_in, int L_out);
    field<mat> computeGradient(mat X, imat y, double lambda);
    double train(mat X, imat y, double lambda, int iterations_num);
    int predict(mat X);
};


NN::NN(Col<int> layers) : Layers(layers) {
    Theta = field<mat>(layers.n_rows-1);
    for (int l = 0; l < layers.n_rows-1; l++) {
        Theta(l) = randInitializeWeights(layers(l), layers(l+1));
    }
}

mat NN::randInitializeWeights(int L_in, int L_out) {
    double epsilon_init = 0.12;
    return (mat(L_out, L_in+1).randu() * (2 * epsilon_init)) - epsilon_init;
}

double NN::train(mat X, imat y, double lambda, int iterations_num) {
    for (int i = 0; i < iterations_num; i++) {
        field<mat> grad = computeGradient(X, y, lambda);
        for (int j = 0; j < Theta.size(); j++) {
            Theta(j) = Theta(j) - grad(j);
        }
        cout <<".";
    }
    return 0;
}

mat NN::sigmoidGradient(mat z) {
    return sigmoid(z) % (1 - sigmoid(z));
}

mat NN::sigmoid(mat z) {
    return 1 / (1 + exp(-z));
}

field<mat> NN::computeGradient(mat X, imat y, double lambda) {
    int m = X.n_rows;
    int L = Theta.size()+1;
    
    field<mat> a(L);
    field<mat> z(L);
    field<mat> d(L);
    field<mat> D(L);
    
    mat Ymatrix(m, num_labels);
    Ymatrix.zeros();
    for (int i = 0; i < m; i++)
        Ymatrix(i, y(i, 0)%10) = 1;
    
    a(0) = X.t();
    for (int i = 1; i < L; i++) {
        int n = a(i-1).n_cols;
        a(i-1) = join_vert(mat(1,n).ones(), a(i-1));
        z(i-1) = join_vert(mat(1,n).ones(), z(i-1));
        z(i) = Theta(i-1) * a(i-1);
        a(i) = sigmoid(z(i));
    }
    
    d(L-1) = a(L-1) - Ymatrix.t();
    for (int i = L-2; i > 0; i--) {
        if(i != L-2)
            d(i+1) = d(i+1).submat(1, 0, d(i+1).n_rows-1, d(i+1).n_cols-1);
        d(i) = (Theta(i).t() * d(i+1)) % sigmoidGradient(z(i));
        if(i == 1)
            d(i) = d(i).submat(1, 0, d(i).n_rows-1, d(i).n_cols-1);
    }
    
    for (int i = 0; i < L-1; i++) {
       // if(i != L-2)
            //d(i+1) = d(i+1).submat(1, 0, d(i+1).n_rows-1, d(i+1).n_cols-1);
        D(i) = d(i+1) * a(i).t();
        D(i) = D(i) / m;
        mat t(Theta(i));
        t.col( 0 ) = mat(t.n_rows,1).zeros();
        t = t * (lambda/m);
        D(i) = D(i) + t;
    }
    
    return D;
}

int NN::predict(mat X) {
    int L = Theta.size()+1;
    field<mat> a(L);
    
    a(0) = X.t();
    for (int i = 1; i < L; i++) {
        int n = a(i-1).n_cols;
        a(i-1) = join_vert(mat(1,n).ones(), a(i-1));
        mat z(Theta(i-1) * a(i-1));
        a(i) = sigmoid(z);
    }
    mat s(a(L-1));
    return a(L-1).index_max();
}




int main(int argc, const char * argv[]) {
    
    // load from file
    mat X;
    X.load("X.txt");
    imat y;
    y.load("y.txt");
    
    int m = X.n_rows;
    
    Col<int> layers;
    layers << input_layer_size << hidden_layer_size << 30 << num_labels;
    
    NN myNet(layers);
    myNet.train(X, y, lambda, 500);
    
    cout << myNet.predict(X.row(251));
    cout << y(251,0);
    return 0;
}






