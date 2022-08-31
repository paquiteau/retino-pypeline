from .base import BaseWorkflowFactory, node_name, getsubid, subid_varname
import os
from dataclasses import dataclass

import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio
import nipype.interfaces.matlab as mlab
import nipype.interfaces.spm as spm
from nipype import IdentityInterface, Function, Node, Workflow

from retino.interfaces.denoise import NoiseStdMap, PatchDenoise, Mask
from retino.interfaces.topup import myTOPUP
from retino.interfaces.motion import RigidMotion

MATLAB_CMD = "matlab -nosplash -nodesktop"

@dataclass
class DenoiseParameters:
    """Denoise Parameters data structure."""
    method: str = None
    patch_shape: int  = 11
    patch_overlap: int = 0
    recombination: str = "weighted" # "center" is also available
    use_phase: bool = False # denoise using complex images.

    @property
    def pretty_name(self):
        if self.method:
            name = self.method
            name += f"_{self.patch_shape}_{self.patch_overlap}_{self.recombination[0]}"
            if self.use_phase:
                name += "_p"
        else:
            name = "noisy"
        return name

    @property
    def pretty_par(self):
        name = f"_{self.patch_shape}_{self.patch_overlap}{self.recombination[0]}"
        if self.use_phase:
            name += "p"
        return name

class PreprocessingWorkflowFactory(BaseWorkflowFactory):
    def __init__(self, base_data_dir, working_dir):
        self.working_dir = working_dir
        self.basedata_dir = base_data_dir

    def _add_realign_node(self):
        # 1. Realign  with SPM
        realign = Node(spm.Realign(), name="realign")
        realign.inputs.separation = 1.0
        realign.inputs.fwhm = 1.0
        realign.inputs.register_to_mean = False
        realign.interface.mlab = mlab.MatlabCommand(
            matlab_cmd=MATLAB_CMD, resource_monitor=False, single_comp_thread=False
        )
        realign.n_procs = 3
        return realign

    def _add_topup_wf(self):
        in_topup = Node(IdentityInterface(fields=["blips", "blip_opposite"]), name="in")
        out_topup = Node(IdentityInterface(fields=["corrected"]), name="out")
        roi_ap = Node(fsl.ExtractROI(t_min=5, t_size=1), name="roi_ap")


        def fsl_merge(in1, in2):
            import nipype.interfaces.fsl as fsl
            merger = fsl.Merge(in_files=[in1, in2], dimension="t")
            results =  merger.run()
            return results.outputs.merged_file

        fsl_merger = Node(Function(inputs_name=["in1","in2"], function=fsl_merge),
                          name="merger")
        # 2.3 Topup Estimation
        topup = Node(myTOPUP(), name="topup")
        topup.inputs.fwhm = 0
        topup.inputs.subsamp = 1
        topup.inputs.out_base = "topup_out"
        topup.inputs.encoding_direction = ["y-", "y"]
        topup.inputs.readout_times = 1.0
        topup.inputs.output_type = "NIFTI"
        # 2.4 Topup correction
        applytopup = Node(fsl.ApplyTOPUP(), name="applytopup")
        applytopup.inputs.in_index = [1]
        applytopup.inputs.method = "jac"
        applytopup.inputs.output_type = "NIFTI"

        topup_wf = Workflow(name="topup", base_dir=self.working_dir)

        topup_wf.connect(
            [(in_topup, roi_ap, [("blips", "in_file")]),
             (in_topup, fsl_merger, [("blip_opposite", "in2")]),
             (roi_ap, fsl_merger, [("roi_file", "in1")]),
             (fsl_merger, topup, [("out", "in_file")]),
             (
                 topup,
                 applytopup,
                 [
                     ("out_fieldcoef", "in_topup_fieldcoef"),
                     ("out_movpar", "in_topup_movpar"),
                     ("out_enc_file", "encoding_file"),
                 ],
             ),
             (in_topup, applytopup, [("blips", "in_files")]),
             (applytopup, out_topup, [("out_corrected", "corrected")])
            ]
        )
        return  topup_wf

    def _add_coregistration_wf(self):
        in_node = Node(IdentityInterface(fields=["func", "anat"]), name="in")
        out_node = Node(IdentityInterface(fields=["coreg_func", "coreg_anat"]), name="out")

        roi_coreg = Node(fsl.ExtractROI(t_min=0, t_size=1), name="roi_coreg")
        roi_coreg.inputs.output_type = "NIFTI"

        coreg = Node(spm.Coregister(), name="coregister")
        coreg.inputs.separation = [1, 1]
        coreg.interface.mlab = mlab.MatlabCommand(
            matlab_cmd=MATLAB_CMD, resource_monitor=False, single_comp_thread=False
        )

        coreg_wf = Workflow(name="coreg", base_dir = self.working_dir)

        coreg_wf.connect(
            [
                (in_node, roi_coreg, [("func", "in_file")]),
                (in_node, coreg, [("anat", "source"),
                                  ("func", "apply_to_files")]),
                (roi_coreg, coreg, [("roi_file", "target")]),
                (coreg, out_node, [("coregistered_files", "coreg_func"),
                                   ("coregistered_source", "coreg_anat")]),
            ]
        )
        return coreg_wf
    def __file_node(self, template):
        files = Node(
            nio.DataGrabber(
                infields=["sub_id"],
                outfields=list(template.keys()),
                base_directory=self.basedata_dir,
                template="*",
                sort_filelist=True,
            ),
            name="selectfiles"
        )
        files.inputs.field_template = template
        files.inputs.templates_args = {k: [["sub_id"]] for k in template.keys()}
        return files


    def _make_inner_wf(self, name, denoise_parameters,  sequence="EPI3D"):
        assert name in ["Clock", "AntiClock"]

        wf = Workflow(name=f"process_{name}", base_dir=self.working_dir)

        in_node = Node(IdentityInterface(fields=["sub_id", "noise_std_map", "mask", "denoise_method"] ), "in")
        out_node = Node(IdentityInterface(fields=["anat", "processed_func", "motion"]), "out")

        template = {
            "anat": "sub_%02i/anat/*_T1.nii",
            "data": f"sub_%02i/func/*{sequence}_{name}wiseTask.nii"
        }

        if "EPI" in sequence:
            template["data_pa"]  = "sub_%02i/func/*1rep_PA.nii"
        if denoise_parameters.use_phase:
            template["data_phase"] = f"sub_%02i/func/*{sequence}_{name}wiseTask_phase.nii"

        files = self.__file_node(template)

        realign = self._add_realign_node()
        coregister = self._add_coregistration_wf()

        connections = [
            # head of workflow
            (in_node, files, [("sub_id", "sub_id")]),
            (files, realign, [("data", "in_files")]),
            # tail of workflow
            (files, coregister, [("anat", "in.anat")]),
            (coregister, out_node, [("out.coreg_func", "processed_func"),
                                    ("out.coreg_anat", "anat")]),
            (realign, out_node, [("realignment_parameters", "motion")])
        ]
        denoise_wf = self._make_noise_workflow(denoise_parameters, extra_name=name)

        connections += [(realign, denoise_wf, [("realigned_files","in.data")]),
                        (in_node, denoise_wf, [("noise_std_map", "in.noise_std_map"),
                                            ("mask", "in.mask"),
                                            ("denoise_method", "in.denoise_method"),
                                            ])]
        if denoise_parameters.use_phase:
            connections += [(realign, denoise_wf,[("realignment_parameters", "in.motion")]),
                            (files, denoise_wf, [("data_phase", "in.data_phase")])]

        if "EPI" in sequence:
            topup_wf = self._add_topup_wf()
            connections += [
                (files, topup_wf, [("data_pa", "in.blip_opposite")]),
                (denoise_wf, topup_wf, [("out.denoised_file", "in.blips")]),
                (topup_wf, coregister, [("out.corrected", "in.func")])
            ]
        else:
            connections += [(realign, coregister, [("realigned_files", "in.func")])]

        wf.connect(connections)

        return wf

    def _make_noise_workflow(self, denoise_parameters, extra_name):
        # Noise workflow:
        # 1. Get the noise Std map
        # 2. Compute a Mask of the brain
        # 3. Realign the phase map
        #
        # 4a. Combine to get the complex image
        # 4b. Denoise with the selected method
        # 4c. Get the magnitude .


        wf = Workflow(node_name("denoising", extra_name), base_dir=self.working_dir)

        input_node = Node(IdentityInterface(["data", "data_phase", "noise_std_map","mask", "motion", "denoise_method"]), "in")
        output_node = Node(IdentityInterface(["denoised_file"]), "out")

        d_node = Node(PatchDenoise(), name="denoise")
        # match input parameters to denoise node interface
        for attr in ["patch_shape", "patch_overlap", "recombination"]:
            setattr(d_node.inputs, attr, getattr(denoise_parameters, attr))

        connections = [
            (input_node, d_node, [("data", "in_file_mag"),
                                  ("noise_std_map", "noise_std_map"),
                                  ("mask", "mask"),
                                  ("denoise_method", "denoise_method")
                                  ]),
            (d_node, output_node, [("denoised_file", "denoised_file")]),
        ]

        if denoise_parameters.use_phase:
            rigid_motion = Node(RigidMotion(), name="rigid_motion")

            connections += [
                (input_node, rigid_motion, [("data_phase", "in_file"),
                                            ("motion", "motion_file")]),
                (rigid_motion, d_node, [("out", "in_file_phase")]),
            ]
        wf.connect(connections)
        return wf

    def _make_predenoise_wf(self, denoise_params, sequence):
        in_node = Node(IdentityInterface(fields=["sub_id"]), name="in")
        out_node = Node(IdentityInterface(fields=["mask", "noise_std_map"]), name="out")
        template = {
            "noise": f"sub_%02i/extra/*{sequence}-0v.nii",
            "data": f"sub_%02i/func/*{sequence}_ClockwiseTask.nii"
        }

        files =  self.__file_node(template)

        noise_map = Node(NoiseStdMap(), name="noise_map")
        noise_map.inputs.block_size = denoise_params.patch_shape
        noise_map.inputs.fft_scale = 100 # Magic Number, needs to be configured elsewhere

        brain_mask = Node(Mask(), name="mask")

        wf = Workflow(name="mask_std")
        wf.connect([(in_node, files, [("sub_id", "sub_id")]),
                    (files, brain_mask, [("data", "in_file")]),
                    (files, noise_map, [("noise", "noise_map_file")]),
                    (brain_mask, out_node, [("mask", "mask")]),
                    (noise_map, out_node, [("noise_std_map", "noise_std_map")]),
                    ])

        return wf
    def build(self, sequence="EPI3D", denoise_method=None, patch_shape=11, patch_overlap=0, use_phase=False, recombination="weighted"):
        """
        Build the preprocessing workflow

        Parameters
        ----------
        sequence: str
        denoise_method: str
        patch_shape: int
        patch_overlap: int
        recombination: str

        Returns
        -------
        nipype.Workflow
        """
        denoise_p = DenoiseParameters(
            method=denoise_method,
            patch_shape=patch_shape,
            patch_overlap=patch_overlap,
            recombination=recombination,
            use_phase=use_phase,
        )

        final_wf = Workflow(name=f"preprocess{denoise_p.pretty_par}", base_dir=self.working_dir)

        in_node = Node(IdentityInterface(fields=["sub_id"] ), "infosource")

        sinker = Node(nio.DataSink(), name="sinker")
        sinker.inputs.base_directory = self.basedata_dir
        sinker.parameterization = False
        sinker.inputs.substitutions = [
            ("rp_sub", "sub"),
            ("rrsub", "sub"),
            ("rsub", "sub"),
            ("_denoise_method_", "")
        ]


        connections = [(
            in_node,
            sinker,
            [
                (("sub_id", getsubid), "container"),
                (("sub_id", subid_varname), "strip_dir"),
            ]
        )]

        if denoise_method:
            wf_mask = self._make_predenoise_wf(denoise_p, sequence)
            proxy_node = Node(IdentityInterface(fields=["mask", "noise_std_map", "denoise_method"]), name="proxy_denoise")
            connections += [(in_node, wf_mask, [("sub_id", "in.sub_id")])]

            connections += [(wf_mask, proxy_node, [("out.mask", "mask"),
                                                 ("out.noise_std_map", "noise_std_map")])]
        out_folder = f"preprocess.{denoise_p.pretty_name}.@"

        for name in ["Clock", "AntiClock"]:
            wf = self._make_inner_wf(name=name, denoise_parameters=denoise_p, sequence=sequence)
            connections += [(in_node, wf, [("sub_id", "in.sub_id")])]
            if denoise_method:
                connections += [(proxy_node, wf, [("mask", "in.mask"),
                                                  ("noise_std_map", "in.noise_std_map"),
                                                  ("denoise_method", "in.denoise_method")])]

            connections += [(wf, sinker, [("out.anat", out_folder + f"anat_{name}"),
                                            ("out.processed_func", out_folder + f"func_{name}"),
                                            ("out.motion", out_folder + f"motion_{name}")])]
        final_wf.connect(connections)
        return final_wf, denoise_p

    @staticmethod
    def run_with_denoiser(wf, sub_ids=None, denoise_methods=None,  plugin=None):
        if sub_ids is not None:
            wf.get_node("infosource").iterables = ("sub_id", sub_ids)
        if not isinstance(denoise_methods, list):
            denoise_methods = list(denoise_methods)

        wf.get_node("proxy_denoise").iterables = ("denoise_method", denoise_methods)
        wf.run(plugin)
