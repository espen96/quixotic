#version 150

#moj_import <light.glsl>

in vec3 Position;
in vec4 Color;
in vec2 UV0;
in ivec2 UV2;
in vec3 Normal;

uniform sampler2D Sampler2;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
uniform vec3 ChunkOffset;
uniform float GameTime;

out float vertexDistance;
out vec4 vertexColor;
out vec2 texCoord0;
out vec4 normal;
out vec4 glpos;
out vec3 gtime;
noperspective out vec3 pos1;
noperspective out vec3 pos2;
noperspective out vec3 pos3;
out vec3 chunkOffset;
out mat4 projInv;
out mat4 projMat;
out vec3 cscale;
out vec4 c1;
out vec4 c2;
out vec4 c3;

vec4 vertexPositions[4] = vec4[] (vec4(- 1, 1, 0, 1), vec4(- 1, - 1, 0, 1), vec4(1, - 1, 0, 1), vec4(1, 1, 0, 1));

mat4 fastInverseProjMat(mat4 projMat) {
return mat4(1.0 / projMat[0][0], 0, 0, 0, 0, 1.0 / projMat[1][1], 0, 0, 0, 0, 0, 1.0 / projMat[3][2], 0, 0, - 1, projMat[2][2] / projMat[3][2]);
}
vec3 encodeFloat24(float val) {
uint sign = val > 0.0 ? 0u : 1u;
uint exponent = uint(log2(abs(val)));
uint mantissa = uint((abs(val) / exp2(float(exponent)) - 1.0) * 131072.0);
return vec3((sign << 7u) | ((exponent + 31u) << 1u) | (mantissa >> 16u), (mantissa >> 8u) & 255u, mantissa & 255u) / 255.0;
}

float decodeFloat24(vec3 raw) {
uvec3 scaled = uvec3(raw * 255.0);
uint sign = scaled.r >> 7;
uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
return (- float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}
vec3 worldToView(vec3 worldPos) {

vec4 pos = vec4(worldPos, 0.0);
pos = ModelViewMat * pos + ModelViewMat[3];

return pos.xyz;
}

void main() {
projInv = mat4(0);
cscale = vec3(0);
c1 = vec4(0);
c2 = vec4(0);
c3 = vec4(0);

vec4 viewPos = ModelViewMat * vec4(Position + ChunkOffset, 1.0);
vertexColor = Color;

mat4 mvm = ModelViewMat ;
vec3 p = Position ;
if(gl_VertexID < 4) {
if(gl_VertexID == 0) {
c1 = viewPos;
} else if(gl_VertexID == 1 || gl_VertexID == 3) {
c2 = viewPos;

} else if(gl_VertexID == 2) {
c3 = viewPos;
}

projInv = fastInverseProjMat(ProjMat);
projMat = ProjMat;
chunkOffset = ChunkOffset;
gl_Position = vertexPositions[gl_VertexID];

p = Position*0;
mvm = ModelViewMat*0;

} else {
gl_Position = vec4(0);
}
pos1 = encodeFloat24((p.x + mvm[3].x));
gtime = encodeFloat24(abs(
    mvm[3].x + mvm[3].y + mvm[3].z + mvm[3].w + 
    mvm[2].x + mvm[2].y + mvm[2].z + mvm[2].w + 
    mvm[1].x + mvm[1].y + mvm[1].z + mvm[1].w +
    mvm[0].x + mvm[0].y + mvm[0].z + mvm[0].w +

    Position.x + Position.y + Position.z

     ));
float temp = (floor(p.y) - 125) + ((mvm[3].y));

pos2 = encodeFloat24(temp);
pos3 = encodeFloat24((p.z) + mvm[3].z);
glpos = gl_Position;
}
