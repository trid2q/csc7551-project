#ifndef _MATRIX_

#define _MATRIX_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <limits>
#include <algorithm>
#include <iostream>

using namespace std;

namespace edu {
namespace tri {
namespace cnn {

template<class T>
class Matrix {
    friend ostream& operator<<(ostream& os, Matrix<float>& src);

    public:
        Matrix() {
            width_ = 0;
            height_ = 0;
            mem_ = NULL;
        }
        Matrix(int w, int h): width_(w), height_(h) {
            assert(w > 0 && h > 0);
            mem_ = new T[width_ * height_];
        }
        Matrix(const Matrix& src) {
            width_ = src.width();
            height_ = src.height();
            mem_ = new T[width_ * height_];
            memcpy(mem_, src.mem_, src.width()*src.height()*sizeof(T));
        }
        
        virtual ~Matrix() {
            delete []mem_;    
        }
	void clear() {
	    memset(mem_, 0, size()*sizeof(T));
	}
        Matrix<T>& operator=(const Matrix<T>& src);

        void randomize() {
            int maxval = width()*height();
            for (int i = 0; i < height(); ++i) {
                for (int j = 0; j < width(); ++j) {
                    //setValue(i, j, T(rand()%10));  // Save this line for testing with integer
		    (*this)(i, j) = 1.0f / maxval * rand() / float( RAND_MAX );                    
                }
            }
        }
        int width() const {
            return width_;
        }
        int height() const {
            return height_;
        }
        Matrix<T>& operator-=(const Matrix<T>& src) {
            for (int i = 0; i < height(); ++i) {
                for (int j = 0; j < width(); ++j) {
                    (*this)(i,j) -= src(i,j);
                }
            }
            return *this;
        }
	int size() const {
	    return width()*height();
	}
        void setValue(int i, int j, T& value) {
            assert(i >= 0 && i < height());
            assert(j >= 0 && j < width());
            mem_[i*width() + j] = value;
        }

        T& value(int i, int j) const {
            assert(i >= 0 && i < height());
            assert(j >= 0 && j < width());
            return mem_[i*width() + j];
        }
        T& operator()(int i, int j) const {
            assert(i >= 0 && i < height());
            assert(j >= 0 && j < width());
            return mem_[i*width() + j];
        }
	void save(ofstream& ofile) {
	    for (int i = 0; i < height(); ++i) {
	        for (int j = 0; j < width(); ++j) {
//		   ofile << (*this)(i,j);
		}
	    }
	}	
	void load(ifstream& ifile) {
	    for (int i = 0; i < height(); ++i) {
	        for (int j = 0; j < width(); ++j) {
//		   ifile >> (*this)(i,j);
		}
	    }
	}	
            
    private:
        unsigned int width_;
        unsigned int height_;
        T* mem_;
};
}
}
}

using namespace edu::tri::cnn; 

#endif
    
