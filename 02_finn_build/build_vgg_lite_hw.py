import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.builder.build_dataflow_steps as build_steps
from qonnx.custom_op.registry import getCustomOp
from finn.util.basic import make_build_dir
import os
import json

# =====================================================================
# Run FINN and Vivado jobs one at a time, to stay within the build
# machine's memory
# =====================================================================
os.environ["NUM_DEFAULT_WORKERS"] = "1"

# =====================================================================
# Custom build step: cap FIFO depths at 512
# ---------------------------------------------------------------------
# step_set_fifo_depths sizes every FIFO by simulation. The deepest ones
# (up to 49,152 words) cost far more logic and memory than the chip can
# spare, so this step, which runs right after it, limits every FIFO to
# 512 words. Shallower FIFOs keep the depth FINN chose. The capped design
# runs on the board without stalling and is bit-exact with the model.
# Note: final_hw_config.json is written by step_set_fifo_depths, i.e.
# before this cap; the applied depths are printed in the build log.
# =====================================================================
def step_smart_cap_fifos(model, cfg):
    print("\n" + "="*60)
    print("CUSTOM STEP: SMART FIFO CAPPING (MAX 512)")
    print("="*60)
    for node in model.graph.node:
        if "FIFO" in node.op_type:
            inst = getCustomOp(node)
            current_depth = int(inst.get_nodeattr("depth"))
            if current_depth > 512:
                inst.set_nodeattr("depth", 512)
                print(f"[FIFO CAP] {node.op_type} '{node.name}' depth {current_depth} -> 512")
    print("="*60 + "\n")
    return model

def main():
    print("--- Starting FINN hardware build ---")

    # 1. Define input files
    model_name = "vgg_lite_int8"
    onnx_file = f"{model_name}.onnx"

    if not os.path.exists(onnx_file):
        print(f"Error: Could not find {onnx_file} in the current directory.")
        return

    # 2. Create output directory
    output_dir = make_build_dir("build_" + model_name + "_bitfile_final")
    print(f"Output files will be generated in: {output_dir}")

    # ==========================================
    # 3. Folding configuration: parallelism and weight storage per layer
    #    cycles per image = (MW / SIMD) x (MH / PE) x output pixels
    # ==========================================
    folding_config = {
        "Defaults": {},

        # conv2 (2,304 weights): small enough to keep its weights inside the block, in LUT RAM
        "MVAU_hls_0": { "PE": 2, "SIMD": 3, "ram_style": "distributed", "mem_mode": "internal_embedded" },

        # conv3 and conv4 (4,608 and 9,216 weights): weights in on-chip BRAM
        "MVAU_hls_1": { "PE": 2, "SIMD": 4, "ram_style": "block", "mem_mode": "internal_decoupled" },
        "MVAU_hls_2": { "PE": 2, "SIMD": 4, "ram_style": "block", "mem_mode": "internal_decoupled" },

        # These two names match no node of the final network; FINN ignores them.
        # They are kept so this script stays identical to the build that produced the bitstream.
        "MVAU_hls_3": { "PE": 2, "SIMD": 4, "ram_style": "block", "mem_mode": "internal_decoupled" },
        "MVAU_rtl_0": { "PE": 2, "SIMD": 2, "ram_style": "block" }
    }

    folding_config_file = os.path.join(output_dir, "custom_folding.json")
    with open(folding_config_file, "w") as f:
        json.dump(folding_config, f, indent=4)
    print(f"Created folding config at: {folding_config_file}")
    # ==========================================

    # 4. Build steps, in order
    my_steps = [
        build_steps.step_qonnx_to_finn,
        build_steps.step_tidy_up,
        build_steps.step_streamline,
        build_steps.step_convert_to_hw,
        build_steps.step_create_dataflow_partition,
        build_steps.step_specialize_layers,
        build_steps.step_target_fps_parallelization,
        build_steps.step_apply_folding_config,
        build_steps.step_minimize_bit_width,
        build_steps.step_generate_estimate_reports,
        build_steps.step_hw_codegen,
        build_steps.step_hw_ipgen,
        build_steps.step_set_fifo_depths,
        step_smart_cap_fifos,
        build_steps.step_create_stitched_ip,
        build_steps.step_measure_rtlsim_performance,
        build_steps.step_out_of_context_synthesis,
        build_steps.step_synthesize_bitfile,
        build_steps.step_make_pynq_driver,
        build_steps.step_deployment_package
    ]

    # 5. Build configuration: PYNQ target, 20 ns clock (50 MHz), full synthesis
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        board="Pynq-Z1",
        synth_clk_period_ns=20.0,

        steps=my_steps,
        auto_fifo_depths=True,
        split_large_fifos=False,

        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        folding_config_file=folding_config_file,

        generate_outputs=[
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.BITFILE,
    build_cfg.DataflowOutputType.PYNQ_DRIVER,
    build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
]
    )

    # 6. Run the build
    print("Loading the ONNX model into FINN and starting the hardware build...")

    try:
        build.build_dataflow_cfg(onnx_file, cfg)
        print("\n--- BITSTREAM GENERATION COMPLETED SUCCESSFULLY! ---")
        print(f"Your final hardware files are ready in: {output_dir}/bitfile")
    except Exception as e:
        print(f"\nBuild failed with error: {e}")

if __name__ == "__main__":
    main()