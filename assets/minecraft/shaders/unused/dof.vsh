#version 150
in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform vec2 InSize;
uniform sampler2D MainSampler;
uniform sampler2D PreviousFrameSampler;


out float near;
out float far;
out float center;
out vec2 texCoord;
out vec2 oneTexel;


// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
vec2 getControl(int index, vec2 screenSize) {
    return vec2(floor(screenSize.x / 2.0) + float(index) * 2.0 + 0.5, 0.5) / screenSize;
}

int intmod(int i, int base) {
    return i - (i / base * base);
}

int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}


float decodeFloat(vec3 ivec) {
    return decodeInt(ivec) / FPRECISION;
}

void main() {
    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);
    gl_Position = vec4(outPos.xy, 0.2, 1.0);
    texCoord = Position.xy / OutSize;

    oneTexel = 1.0 / InSize;
    vec2 start = getControl(0, OutSize);
    vec2 inc = vec2(2.0 / OutSize.x, 0.0);


    mat4 ProjMat = mat4(tan(decodeFloat(texture(MainSampler, start + 3.0 * inc).xyz)), decodeFloat(texture(MainSampler, start + 6.0 * inc).xyz), 0.0, 0.0,
                        decodeFloat(texture(MainSampler, start + 5.0 * inc).xyz), tan(decodeFloat(texture(MainSampler, start + 4.0 * inc).xyz)), decodeFloat(texture(MainSampler, start + 7.0 * inc).xyz), decodeFloat(texture(MainSampler, start + 8.0 * inc).xyz),
                        decodeFloat(texture(MainSampler, start + 9.0 * inc).xyz), decodeFloat(texture(MainSampler, start + 10.0 * inc).xyz), decodeFloat(texture(MainSampler, start + 11.0 * inc).xyz),  decodeFloat(texture(MainSampler, start + 12.0 * inc).xyz),
                        decodeFloat(texture(MainSampler, start + 13.0 * inc).xyz), decodeFloat(texture(MainSampler, start + 14.0 * inc).xyz), decodeFloat(texture(MainSampler, start + 15.0 * inc).xyz), 0.0);


    near = PROJNEAR;
    far = ProjMat[3][2] * PROJNEAR / (ProjMat[3][2] + 2.0 * PROJNEAR);

}
