#include "Matrix.h"

#include <iostream>
#include <iomanip>

using namespace std;

namespace edu {
namespace tri {
namespace cnn {

ostream& operator<<(ostream & os, Matrix<float>& src) {
    for (int i = 0; i < src.height(); ++i) {
        os << "Row " << i << ": ";
        for (int j = 0; j < src.width(); ++j) {
            if (j > 0) os << ", ";
	    os << setprecision(2) << src.value(i, j);
        }
        os << endl;
    }
    return os;
}
template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& src) {
    delete []mem_;    
    width_ = src.width();
    height_ = src.height();
    mem_ = new T[width_ * height_];
    memcpy(mem_, src.mem_, src.width()*src.height()*sizeof(T));
    return *this;
}

}
}
}
