#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <caffe/openpose/layers/oPDataLayer.hpp>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
namespace py = pybind11;

class OPCaffe{
public:
    std::shared_ptr<OPDataLayer<float>> dataLayer;
    std::vector<Blob<float>*> bottom, top;

    OPCaffe(py::dict d){
        LayerParameter param;
        param.mutable_data_param()->set_batch_size(py::int_(d["batch_size"]));
        param.mutable_data_param()->set_backend(DataParameter::LMDB);
        param.mutable_op_transform_param()->set_stride(py::int_(d["stride"]));
        param.mutable_op_transform_param()->set_max_degree_rotations(py::str(d["max_degree_rotations"]));
        param.mutable_op_transform_param()->set_crop_size_x(py::int_(d["crop_size_x"]));
        param.mutable_op_transform_param()->set_crop_size_y(py::int_(d["crop_size_y"]));
        param.mutable_op_transform_param()->set_center_perterb_max(py::float_(d["center_perterb_max"]));
        param.mutable_op_transform_param()->set_center_swap_prob(py::float_(d["center_swap_prob"]));
        param.mutable_op_transform_param()->set_scale_prob(py::float_(d["scale_prob"]));
        param.mutable_op_transform_param()->set_scale_mins(py::str(d["scale_mins"]));
        param.mutable_op_transform_param()->set_scale_maxs(py::str(d["scale_maxs"]));
        param.mutable_op_transform_param()->set_target_dist(py::float_(d["target_dist"]));
        param.mutable_op_transform_param()->set_number_max_occlusions(py::str(d["number_max_occlusions"]));
        param.mutable_op_transform_param()->set_sigmas(py::str(d["sigmas"]));
        param.mutable_op_transform_param()->set_models(py::str(d["models"]));
        param.mutable_op_transform_param()->set_sources(py::str(d["sources"]));
        param.mutable_op_transform_param()->set_probabilities(py::str(d["probabilities"]));
        param.mutable_op_transform_param()->set_source_background(py::str(d["source_background"]));
        param.mutable_op_transform_param()->set_normalization(py::int_(d["normalization"]));
        param.mutable_op_transform_param()->set_add_distance(py::int_(d["add_distance"]));

        dataLayer = std::shared_ptr<OPDataLayer<float>>(new OPDataLayer<float>(param));

        bottom = {new Blob<float>{1,1,1,1}};
        top = {new Blob<float>{1,1,1,1}, new Blob<float>{1,1,1,1}};
        dataLayer->DataLayerSetUp(bottom, top);
    }

    void load(Batch<float>& batch){
        batch.data_.Reshape(top[0]->shape());
        batch.label_.Reshape(top[1]->shape());
        dataLayer->load_batch(&batch);
    }
};

PYBIND11_MODULE(opcaffe, m){
    py::class_<OPCaffe>(m, "OPCaffe")
        .def(py::init<py::dict>())
        .def("load", &OPCaffe::load)
        ;

    py::class_<caffe::Batch<float>, std::shared_ptr<caffe::Batch<float>>>(m, "Batch")
        .def(py::init<>())
        .def_readonly("data", &caffe::Batch<float>::data_)
        .def_readonly("label", &caffe::Batch<float>::label_)
        ;

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

// Numpy - caffe::Blob<float> interop
namespace pybind11 { namespace detail {

template <> struct type_caster<caffe::Blob<float>> {
    public:

        PYBIND11_TYPE_CASTER(caffe::Blob<float>, _("numpy.ndarray"));

        // Cast numpy to op::Array<float>
        bool load(handle src, bool imp)
        {
            throw std::runtime_error("Not implemented");
        }

        // Cast op::Array<float> to numpy
        static handle cast(const caffe::Blob<float> &m, return_value_policy, handle defval)
        {
            std::string format = format_descriptor<float>::format();
            return array(buffer_info(
                m.pseudo_cpu_data(),    /* Pointer to buffer */
                sizeof(float),          /* Size of one scalar */
                format,                 /* Python struct-style format descriptor */
                m.shape().size(),       /* Number of dimensions */
                m.shape(),              /* Buffer dimensions */
                m.stride()              /* Strides (in bytes) for each index */
                )).release();
        }

    };
}} // namespace pybind11::detail


#endif  // USE_OPENCV
