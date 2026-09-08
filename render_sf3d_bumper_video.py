"""Render embedded GLB materials without OBJ conversion. 12s / 720p / 30fps."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

# Existing optional local tools; normal pip installation also works.
LOCAL_TOOLS = Path(__file__).resolve().parent.parent / ".video_tools"
if LOCAL_TOOLS.is_dir():
    sys.path.insert(0, str(LOCAL_TOOLS))
import numpy as np
import trimesh
from PIL import Image, ImageDraw

VERTEX = """
#version 330
uniform mat4 mvp;
in vec3 position; in vec3 normal; in vec2 uv;
out vec3 p; out vec3 n; out vec2 tex;
void main(){ p=position; n=normal; tex=uv; gl_Position=mvp*vec4(position,1.0); }
"""
FRAGMENT = """
#version 330
uniform sampler2D albedo; uniform sampler2D mr;
uniform vec4 base_factor; uniform float rough_factor; uniform float metal_factor;
uniform vec3 eye; uniform int alpha_mode; uniform float alpha_cutoff;
in vec3 p; in vec3 n; in vec2 tex; out vec4 frag;
const float PI=3.14159265;
vec3 light(vec3 N,vec3 V,vec3 L,vec3 base,float metal,float rough,vec3 power){
 vec3 H=normalize(V+L); float nl=max(dot(N,L),0.0),nv=max(dot(N,V),0.001);
 float nh=max(dot(N,H),0.0),vh=max(dot(V,H),0.0);
 float a=rough*rough,a2=a*a,d=nh*nh*(a2-1.0)+1.0;
 float D=a2/(PI*d*d+0.00001),k=(rough+1.0)*(rough+1.0)/8.0;
 float G=(nv/(nv*(1.0-k)+k))*(nl/(nl*(1.0-k)+k));
 vec3 f0=mix(vec3(0.04),base,metal),F=f0+(1.0-f0)*pow(1.0-vh,5.0);
 return ((1.0-F)*(1.0-metal)*base/PI+D*G*F/(4.0*nv*nl+0.001))*power*nl;
}
void main(){
 vec4 sample_color=texture(albedo,tex);
 float alpha=sample_color.a*base_factor.a;
 if(alpha_mode==1 && alpha<alpha_cutoff) discard;
 vec3 base=pow(sample_color.rgb,vec3(2.2))*base_factor.rgb;
 vec2 props=texture(mr,tex).gb;
 float rough=clamp(props.x*rough_factor,0.12,1.0),metal=props.y*metal_factor;
 vec3 N=normalize(n),V=normalize(eye-p); if(!gl_FrontFacing) N=-N;
 vec3 col=base*(0.40+0.20*max(N.y,0.0));
 col+=light(N,V,normalize(vec3(-2,3,4)),base,metal,rough,vec3(4.5,4.4,4.2));
 col+=light(N,V,normalize(vec3(3,1,2)),base,metal,rough,vec3(2.4,2.6,3.0));
 col+=light(N,V,normalize(vec3(1,3,-3)),base,metal,rough,vec3(4.0));
 col=col/(col+vec3(1.0)); frag=vec4(pow(col,vec3(1.0/2.2)),1.0);
}
"""


def load_instances(path):
    scene = trimesh.load(path, force="scene", process=False)
    instances = []
    for node in scene.graph.nodes_geometry:
        transform, key = scene.graph[node]
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(transform)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Empty/nonfinite GLB instance")
        instances.append(mesh)
    if not instances:
        raise ValueError("No GLB geometry")
    bounds = np.array([m.bounds for m in instances])
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    extent = (high-low).max()
    if extent <= 0:
        raise ValueError("Degenerate GLB")
    for mesh in instances:
        mesh.apply_translation(-(low+high)/2)
        mesh.apply_scale(2/extent)
    return instances


def verify_video(path, width=1280, height=720, frames=360, fps=30):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    count = 0
    try:
        metadata = next(reader)
        if tuple(metadata["size"]) != (width, height) or abs(metadata["fps"]-fps) > .01:
            raise ValueError("Unexpected video size or fps")
        for frame in reader:
            if len(frame) != width*height*3:
                raise ValueError("Incomplete frame")
            if count == frames//2:
                Image.frombytes("RGB", (width, height), frame).save(path.parent / "video_midpoint.jpg")
            count += 1
    finally:
        reader.close()
    if count != frames:
        raise ValueError(f"Expected {frames} frames, decoded {count}")
    return {"decoded_frames": count, "fps": fps, "duration_seconds": count/fps,
            "resolution": [width, height]}


def render_video(source, out, preview=False, front_angle=0, backend=None):
    import moderngl
    import imageio_ffmpeg
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output must be empty; choose a new directory")
    meshes = load_instances(source)
    out.mkdir(parents=True, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=330, **({"backend": backend} if backend else {}))
    try:
        program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        draws = []
        for mesh in meshes:
            material = getattr(mesh.visual, "material", None)
            if material is None:
                raise ValueError("GLB material missing")
            mode = getattr(material, "alphaMode", None) or "OPAQUE"
            if mode == "BLEND":
                raise ValueError("BLEND material needs transparent sorting; use a full glTF renderer")
            uv = getattr(mesh.visual, "uv", None)
            base = getattr(material, "baseColorTexture", None)
            if base is not None and (uv is None or not np.isfinite(uv).all()):
                raise ValueError("Textured material requires finite UVs")
            if uv is None:
                uv = np.zeros((len(mesh.vertices), 2))
            data = np.column_stack([mesh.vertices, mesh.vertex_normals, uv]).astype("f4")
            vao = ctx.vertex_array(program, [(ctx.buffer(data.tobytes()), "3f 3f 2f", "position", "normal", "uv")],
                                   ctx.buffer(np.asarray(mesh.faces, dtype="i4").tobytes()))
            textures = []
            for im in (base, getattr(material, "metallicRoughnessTexture", None)):
                im = (im if im is not None else Image.new("RGBA", (1, 1), "white")).convert("RGBA")
                im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                texture = ctx.texture(im.size, 4, im.tobytes())
                texture.build_mipmaps()
                textures.append(texture)
            factor = getattr(material, "baseColorFactor", None)
            factor = np.asarray(factor, dtype=float)/255 if factor is not None else np.ones(4)
            rough = getattr(material, "roughnessFactor", None)
            metal = getattr(material, "metallicFactor", None)
            cutoff = getattr(material, "alphaCutoff", None)
            draws.append((vao, textures, factor, 1 if rough is None else rough,
                          1 if metal is None else metal, mode, .5 if cutoff is None else cutoff))
        W, H = 1280, 720
        fbo = ctx.simple_framebuffer((W, H), components=4)
        fbo.use()
        ctx.enable(moderngl.DEPTH_TEST)
        program["albedo"].value, program["mr"].value = 0, 1
        # Bounding sphere framing prevents clipping even with rotated multi-part scenes.
        radius = max(np.linalg.norm(m.vertices, axis=1).max() for m in meshes)
        distance = max(3.25, radius/math.sin(math.radians(33)/2)*1.12)

        def frame(angle):
            theta = math.radians(angle + front_angle)
            eye = np.array([distance*math.sin(theta), .42, distance*math.cos(theta)])
            z = eye/np.linalg.norm(eye)
            x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            view = np.eye(4); view[:3, :3] = [x, y, z]; view[:3, 3] = -view[:3, :3]@eye
            projection = np.zeros((4, 4)); f = 1/math.tan(math.radians(33)/2)
            projection[0, 0] = f/(W/H); projection[1, 1] = f
            projection[2, 2] = -(100+.1)/(100-.1); projection[2, 3] = -2*100*.1/(100-.1)
            projection[3, 2] = -1
            program["mvp"].write((projection@view).T.astype("f4").tobytes())
            program["eye"].value = tuple(eye)
            fbo.clear(.91, .925, .94, 1)
            for vao, textures, factor, rough, metal, mode, cutoff in draws:
                for unit, texture in enumerate(textures):
                    texture.use(unit)
                program["base_factor"].value = tuple(factor)
                program["rough_factor"].value = rough; program["metal_factor"].value = metal
                program["alpha_mode"].value = int(mode == "MASK"); program["alpha_cutoff"].value = cutoff
                vao.render()
            im = Image.frombytes("RGB", (W, H), fbo.read(components=3, alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            ImageDraw.Draw(im).text((48, 32), "STABLE FAST 3D / ORIGINAL PHOTO", fill=(32, 42, 53))
            return im

        sheet = Image.new("RGB", (1280, 720))
        for i, angle in enumerate((0, 90, 180, 270)):
            im = frame(angle); im.thumbnail((640, 360)); sheet.paste(im, ((i%2)*640, (i//2)*360))
        sheet.save(out / "angle_check.jpg", quality=94)
        frame(0).save(out / "bumper_preview.jpg", quality=95)
        report = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "renderer": ctx.info["GL_RENDERER"], "front_angle": front_angle,
                  "instances": len(meshes), "faces": sum(len(m.faces) for m in meshes),
                  "camera_distance": float(distance), "status": "preview_only",
                  "material_scope": "base color, metallic/roughness, OPAQUE/MASK; no normal/occlusion/emissive maps"}
        if not preview:
            path = out / "bumper_turntable.mp4"
            writer = imageio_ffmpeg.write_frames(str(path), (W, H), fps=30, codec="libx264", quality=8,
                        macro_block_size=16, ffmpeg_log_level="error", output_params=["-movflags", "+faststart"])
            writer.send(None)
            try:
                for index in range(360):
                    t = max(0, min(1, (index-30)/299))
                    writer.send(frame(360*t*t*(3-2*t)).tobytes())
                    if index%60 == 0:
                        print(f"Frame {index}/360", flush=True)
            finally:
                writer.close()
            report.update(verify_video(path), status="video_verified")
        (out / "render_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        ctx.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--front-angle", type=float, default=0)
    ap.add_argument("--gl-backend", choices=["egl", "x11"])
    args = ap.parse_args()
    print(json.dumps(render_video(args.source, args.out, args.preview, args.front_angle, args.gl_backend), indent=2))
