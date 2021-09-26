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
out vec3 chunkOffset;
out mat4 projInv;
out mat4 projMat;
out vec3 cscale;
out vec4 c1;
out vec4 c2;
out vec4 c3;

vec4 vertexPositions[4] = vec4[](
    vec4(-1, 1, 0, 1),    
    vec4(-1, -1, 0, 1),    
    vec4(1, -1, 0, 1),    
    vec4(1, 1, 0, 1)    
);

mat4 fastInverseProjMat(mat4 projMat) {
    return mat4(
        1.0 / projMat[0][0], 0, 0, 0,
        0, 1.0 / projMat[1][1], 0, 0,
        0, 0, 0, 1.0 / projMat[3][2],
        0, 0, -1, projMat[2][2] / projMat[3][2]
    );
}

void main() {
    projInv = mat4(0);
    cscale = vec3(0);
    c1 = vec4(0);
    c2 = vec4(0);
    c3 = vec4(0);

    vec4 viewPos = ModelViewMat * vec4(Position + ChunkOffset, 1.0);
    vertexColor = Color ;


    


    if (gl_VertexID < 4 ) {
        if (gl_VertexID == 0) {
            c1 = viewPos;
        } else if (gl_VertexID == 1 || gl_VertexID == 3) {
            c2 = viewPos;
        } else if (gl_VertexID == 2) {
            c3 = viewPos;
        }
        projInv = fastInverseProjMat(ProjMat);
        projMat = ProjMat;
        chunkOffset = ChunkOffset;
        gl_Position = vertexPositions[gl_VertexID];
    } else {
        gl_Position = ProjMat * viewPos;
    }


    glpos = gl_Position;
}
