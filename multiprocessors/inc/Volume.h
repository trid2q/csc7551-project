#ifndef _VOLUME_

#define _VOLUME_

#include <vector>
#include "Matrix.h"
//#include "Filter.h"

namespace edu {
namespace tri {
namespace cnn {

//class Filter;

template<class T>
class Volume {
    friend ostream& operator<<(ostream& os, Volume<float>& src);

    public:
        Volume() {
        }
        Volume(const Volume& src) {
	   for (int d = 0; d < src.depth(); ++d) {
		Matrix<T>* m = new Matrix<T>(*(src.get(d)));
		matrices_.push_back(m);
	   } 
	}
        Volume(int width, int depth) {
            assert(width > 0 && depth > 0);
            for (int k = 0; k < depth; ++k) {
                matrices_.push_back(new Matrix<T>(width,width));
            }
        }
        Volume(int width, int height, int depth) {
            assert(width > 0 && height > 0 && depth > 0);
            for (int k = 0; k < depth; ++k) {
                matrices_.push_back(new Matrix<T>(width,height));
            }
        }
        virtual ~Volume() {
            for (int i = 0; i < matrices_.size(); ++i) {
                delete matrices_[i];
            }
        }
        void randomize() {
            for (int i = 0; i < depth(); ++i) {
                matrices_[i]->randomize();
            }
        }
        Matrix<float>* get(int d) const {
	    return matrices_[d];
	}
	Volume<T>& operator-=(const Volume<T>& other) {
	    for (int d = 0; d < depth(); ++d) {
		(*this)[d] -= other[d];
	    }
	    return *this;
	}
        Matrix<T>& operator[](int d) const {
            return *(matrices_[d]);
        }
        int width() {
            return matrices_[0]->width();
        }
        int height() {
            return matrices_[0]->height();
        }
        int depth() const {
            return matrices_.size();
        }
        void clear() {
            for (int d = 0; d < depth(); ++d) {
                matrices_[d]->clear();
            }
        }
        void add(Matrix<float>* matrix) {
            assert(matrix);
            matrices_.push_back(matrix);
        }
        int size() {
	    return width()*height()*depth();
        }
        void setValue(int i, int j, int d, T& val) {
	    assert(d < matrices_.size());
            assert(i < height());
	    assert(j < width());
	    matrices_[d]->setValue(i, j, val);
        }
        T& value(int i, int j, int d) const {
	    return matrices_[d]->value(i,j);
        }
        T& operator()(int i, int j, int d) const {
	    return matrices_[d]->value(i,j);
        }
	void save(ofstream& ofile) {
	    for (int i = 0; i < depth(); ++i) {
	        (*this)[i].save(ofile);
	   }
	}
	void load(ifstream& ifile) {
	    for (int i = 0; i < depth(); ++i) {
	        (*this)[i].load(ifile);
	   }
	}
	
    private:
        vector<Matrix<T>*> matrices_;
};

}
}
}

using namespace edu::tri::cnn;
 
#endif
    
