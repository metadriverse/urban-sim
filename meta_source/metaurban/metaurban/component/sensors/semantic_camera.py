from panda3d.core import RenderState, LightAttrib, ColorAttrib, ShaderAttrib, TextureAttrib, FrameBufferProperties

from metaurban.component.sensors.base_camera import BaseCamera
from metaurban.constants import CamMask
from metaurban.constants import Semantics, CameraTagStateKey


class SemanticCamera(BaseCamera):
    CAM_MASK = CamMask.SemanticCam

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        buffer_props = FrameBufferProperties()
        buffer_props.set_rgba_bits(8, 8, 8, 8)
        buffer_props.set_depth_bits(8)
        buffer_props.set_force_hardware(True)
        buffer_props.set_multisamples(16)
        buffer_props.set_srgb_color(False)
        buffer_props.set_stereo(False)
        buffer_props.set_stencil_bits(0)
        super(SemanticCamera, self).__init__(engine, cuda, buffer_props)

    def _setup_effect(self):
        """
        Use tag to apply color to different object class
        Returns: None

        """
        # setup camera
        cam = self.get_cam().node()
        cam.setTagStateKey(CameraTagStateKey.Semantic)
        ts = [v for v, m in vars(Semantics).items() if not (v.startswith('_') or callable(m))]
        for t in [v for v, m in vars(Semantics).items() if not (v.startswith('_') or callable(m))]:
            label, c = getattr(Semantics, t)
            if label == Semantics.TERRAIN.label:
                from metaurban.engine.core.terrain import Terrain
                cam.setTagState(
                    label, Terrain.make_render_state(self.engine, "terrain.vert.glsl", "terrain_semantics.frag.glsl")
                )
            else:
                if label in [Semantics.PEDESTRIAN.label]:#, Semantics.EDOG.label, Semantics.EROBOT.label]:
                    # PZH: This is a workaround fix to make pedestrians animated.
                    cam.setTagState(
                        label,
                        RenderState.make(
                            # ShaderAttrib.makeOff(),
                            LightAttrib.makeAllOff(),
                            TextureAttrib.makeOff(),
                            ColorAttrib.makeFlat((c[0] / 255, c[1] / 255, c[2] / 255, 1)),
                            # ColorAttrib.makeFlat((c[0], c[1], c[2], 1)),
                            1
                        )
                    )

                else:
                    cam.setTagState(
                        label,
                        RenderState.make(
                            ShaderAttrib.makeOff(), LightAttrib.makeAllOff(), TextureAttrib.makeOff(),
                            ColorAttrib.makeFlat((c[0] / 255, c[1] / 255, c[2] / 255, 1)), 1
                        )
                    )