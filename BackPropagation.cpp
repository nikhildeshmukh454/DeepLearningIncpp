#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(const vector<int>& layer_dims) {
        initialize_parameters(layer_dims);
    }

    void forward_propagation(const vector<vector<double>>& X, vector<vector<double>>& A1, vector<double>& y_hat) {
        A1 = X;
        A = X;
        
        for (int l = 1; l < L; ++l) {
            vector<vector<double>> A_prev = A;
            vector<double> Z;

            for (int i = 0; i < W[l-1].size(); ++i) {
                double z = 0;
                for (int j = 0; j < W[l-1][i].size(); ++j) {
                    z += W[l-1][i][j] * A_prev[j][0];
                }
                Z.push_back(z + b[l-1][i][0]);
            }
            A.clear();
            for (auto z : Z) {
                vector<double> a = {z};
                A.push_back(a);
            }
        }
        y_hat = A[0];
    }

    void update_parameters(double y, const vector<double>& y_hat, const vector<vector<double>>& A1, const vector<vector<double>>& X) {
        double alpha = 0.001;
        double diff = y - y_hat[0];

        W[1][0][0] += alpha * 2 * diff * A1[0][0];
        W[1][1][0] += alpha * 2 * diff * A1[1][0];
        b[1][0][0] += alpha * 2 * diff;

        W[0][0][0] += alpha * 2 * diff * W[1][0][0] * X[0][0];
        W[0][0][1] += alpha * 2 * diff * W[1][0][0] * X[1][0];
        b[0][0][0] += alpha * 2 * diff * W[1][0][0];

        W[0][1][0] += alpha * 2 * diff * W[1][1][0] * X[0][0];
        W[0][1][1] += alpha * 2 * diff * W[1][1][0] * X[1][0];
        b[0][1][0] += alpha * 2 * diff * W[1][1][0];
    }

    void train(const vector<vector<vector<double>>>& X, const vector<double>& y, int epochs) {
        for (int i = 0; i < epochs; ++i) {
            double loss = 0;
            for (int j = 0; j < X.size(); ++j) {
                vector<vector<double>> A1;
                vector<double> y_hat;
                forward_propagation(X[j], A1, y_hat);
                update_parameters(y[j], y_hat, A1, X[j]);
                loss += pow(y[j] - y_hat[0], 2);
            }
            cout << "Epoch - " << i + 1 << " Loss - " << loss / X.size() << endl;
        }
    }

private:
    void initialize_parameters(const vector<int>& layer_dims) {
        L = layer_dims.size();
        for (int l = 1; l < L; ++l) {
            vector<vector<double>> w(layer_dims[l], vector<double>(layer_dims[l-1], 0.1));
            W.push_back(w);
            vector<vector<double>> b_vec(layer_dims[l], vector<double>(1, 0));
            b.push_back(b_vec);
        }
    }

    vector<vector<vector<double>>> W;
    vector<vector<vector<double>>> b;
    vector<vector<double>> A;
    int L;
};

int main() {
    vector<int> layer_dims = {2, 2, 1};
    NeuralNetwork nn(layer_dims);

    // Example data, you should replace this with your actual data
    vector<vector<vector<double>>> X = {{{3.5, 2.1}}, {{3.7, 2.3}}, {{4.0, 2.5}}};
    vector<double> y = {6.0, 7.0, 8.0};

    int epochs = 5;
    nn.train(X, y, epochs);

    return 0;
}
