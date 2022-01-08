#version 150

#moj_import <light.glsl>
#moj_import <voxelization.glsl>
in vec3 Position;
in vec4 Color;
in vec2 UV0;
in ivec2 UV2;
in vec3 Normal;
uniform sampler2D Sampler0;
uniform sampler2D Sampler2;
uniform float GameTime;
uniform vec2 ScreenSize;
uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
uniform vec3 ChunkOffset;
out float vertexDistance;

out float lmx;
out float lmy;
out vec4 vertexColor;
out vec4 vertexColor2;
noperspective out vec3 test;
out vec2 texCoord0;
out vec2 texCoord2;
out float dataFace;
out vec4 glpos;
out mat4 ProjMat2;
const vec2 COPRIMES = vec2(2, 3);
out vec4 normal;
vec2 halton(int index) {
vec2 f = vec2(1);
vec2 result = vec2(0);
vec2 ind = vec2(index);

while(ind.x > 0.0 && ind.y > 0.0) {
f /= COPRIMES;
result += f * mod(ind, COPRIMES);
ind = floor(ind / COPRIMES);
}
return result;
}
const vec2[] OFFSETS = vec2[](
    vec2(0, 0),
    vec2(1, 0),
    vec2(1, 1),
    vec2(0, 1)
);

vec2 calculateJitter() {
return (halton(int(mod((GameTime * 3.0) * 24000.0, 128))) - 0.5) / 1024.0;
}

void main() {
vec3 position = Position + ChunkOffset;
float animation = GameTime * 4000.0;
test = textureLod(Sampler0, UV0, 100).rgb;

    if (distance(test.rgb, vec3(1, 0, 1)) < 0.01) {
        if (Normal.y > 0) {
            // Data face used for voxelization
            dataFace = 1.0;
            bool inside;
            // TODO: Add gametime
            ivec2 pixel = positionToPixel(floor(Position + floor(ChunkOffset)), ScreenSize, inside, 0);
            if (!inside) {
                gl_Position = vec4(5, 5, 0, 1);
                return;
            }
            gl_Position = vec4(
                (vec2(pixel) + OFFSETS[imod(gl_VertexID, 4)]) / ScreenSize * 2.0 - 1.0,
                -1,
                1
            );
            //gl_Position = ProjMat * ModelViewMat * (pos + vec4(0, 0.2, 0, 0));
            vertexColor = vec4(floor(Position.xz) / 16, 0, 1);
        } else {
            // Data face used for chunk offset storage
            gl_Position = vec4(
                OFFSETS[imod(gl_VertexID, 4)] * vec2(3, 1) / ScreenSize * 2.0 - 1.0,
                -1,
                1
            );
            dataFace = 2.0;
        }
    } else {
        dataFace = 0.0;
        float xs = 0.0;
        float zs = 0.0;

        texCoord0 = UV0;
        vertexColor2 = Color * minecraft_sample_lightmap2(Sampler2, UV2);

        lmx = clamp((float(UV2.y) / 255), 0, 1);
        lmy = clamp((float(UV2.x) / 255), 0, 1);
        ProjMat2 = ProjMat;

        gl_Position = ProjMat * ModelViewMat * (vec4(position, 1.0) + vec4(xs / 32.0, 0.0, zs / 32.0, 0.0) + vec4(calculateJitter() * 0.0, 0, 0));


        vertexDistance = length((ModelViewMat * vec4(Position + ChunkOffset, 1.0)).xyz);
        vertexColor = Color;
        texCoord0 = UV0;
        normal = ProjMat * ModelViewMat * vec4(Normal, 0.0);

    }


glpos = gl_Position;
}
