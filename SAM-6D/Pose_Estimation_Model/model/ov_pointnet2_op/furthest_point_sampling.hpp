#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
//! [op:common_include]

//! [op:header]
namespace TemplateExtension {

class FurthestPointSampling : public ov::op::Op {
public:
    OPENVINO_OP("FurthestPointSampling");

    FurthestPointSampling() = default;
    FurthestPointSampling(const ov::Output<ov::Node>& xyz, const ov::Output<ov::Node>& npoint);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
//! [op:header]

}  // namespace TemplateExtension