#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "furthest_point_sampling.hpp"
#include "gather_operation.hpp"
#include "three_nn.hpp"
#include "three_interpolate.hpp"
#include "cylinder_query.hpp"
#include "ball_query.hpp"
#include "grouping_operation.hpp"
#include "custom_svd.hpp"
#include "custom_det.hpp"
#include "custom_searchsorted.hpp"
#include "custom_debug_node.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<TemplateExtension::FurthestPointSampling>>(),
        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::FurthestPointSampling>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::GatherOperation>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::GatherOperation>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::ThreeNN>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::ThreeNN>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::ThreeInterpolate>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::ThreeInterpolate>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::GroupingOperation>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::GroupingOperation>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::BallQuery>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::BallQuery>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CylinderQuery>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CylinderQuery>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSVD>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSVD>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomDet>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomDet>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSearchSorted>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSearchSorted>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomDebugNode>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomDebugNode>>(),
    }));
//! [ov_extension:entry_point]
// clang-format on