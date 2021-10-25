#version 150

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D temporals3Sampler;
out vec2 texCoord;
out vec2 texCoord2;
out vec2 oneTexel;
out vec3 sunDir;
out vec4 fogcol;
out vec4 fogColor;
out vec4 rain;
out float GameTime;
out vec4 skycol;
out mat4 gbufferModelViewInverse;
out mat4 gbufferModelView;
out mat4 gbufferProjection;
out mat4 gbufferProjectionInverse;
out float near;
out float far;
out float cosFOVrad;
out float tanFOVrad;
uniform vec2 InSize;
uniform float FOV;
out float aspectRatio;
out mat4 gbPI;
out mat4 gbP;
out vec4 fogColor2;
out vec3 flareColor;
out vec3 sunColor;

uniform float Time;

vec3 rodSample(vec2 Xi)
{
	float r = sqrt(1.0f - Xi.x*Xi.y);
    float phi = 2 * 3.14159265359 * Xi.y;

    return normalize(vec3(cos(phi) * r, sin(phi) * r, Xi.x)).xzy;
}
//Low discrepancy 2D sequence, integration error is as low as sobol but easier to compute : http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}
// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define FPRECISION 4000000.0
#define PROJNEAR 0.05

vec2 getControl(int index, vec2 screenSize) {
    return vec2(floor(screenSize.x / 2.0) + float(index) * 2.0 + 0.5, 0.5) / screenSize;
}

int intmod(int i, int base) {
    return i - (i / base * base);
}

vec3 encodeInt(int i) {
    int s = int(i < 0) * 128;
    i = abs(i);
    int r = intmod(i, 256);
    i = i / 256;
    int g = intmod(i, 256);
    i = i / 256;
    int b = intmod(i, 128);
    return vec3(float(r) / 255.0, float(g) / 255.0, float(b + s) / 255.0);
}

int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}

vec3 encodeFloat(float i) {
    return encodeInt(int(i * FPRECISION));
}

float decodeFloat(vec3 ivec) {
    return decodeInt(ivec) / FPRECISION;
}
vec2 tapLocation(int sampleNumber,int nb, float nbRot,float jitter)
{
    float alpha = float(sampleNumber+jitter)/nb;
    float angle = (jitter+alpha) * (nbRot * 6.28);

    float ssR = alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
}


const float pi = 3.141592653589793238462643383279502884197169;

vec2 sphereToCarte(vec3 dir) {
    float lonlat = atan(-dir.x, -dir.z);
    return vec2(lonlat * (0.5/pi) +0.5,0.5*dir.y+0.5);
}

vec3 skyFromTex(vec3 pos,sampler2D sampler){
	vec2 p = sphereToCarte(pos);
	return texture(sampler,p*oneTexel*256.+vec2(18.5,1.5)*oneTexel).rgb;
}
void main() {



    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);
    gl_Position = vec4(outPos.xy, 0.2, 1.0);
    texCoord = Position.xy / OutSize;
    oneTexel = 1.0 / OutSize;

    //simply decoding all the control data and constructing the sunDir, ProjMat, ModelViewMat

    vec2 start = getControl(0, OutSize);
    vec2 inc = vec2(2.0 / OutSize.x, 0.0);


    // ProjMat constructed assuming no translation or rotation matrices applied (aka no view bobbing).
    mat4 ProjMat = mat4(tan(decodeFloat(texture(DiffuseSampler, start + 3.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 5.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 9.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 13.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 6.0 * inc).xyz), tan(decodeFloat(texture(DiffuseSampler, start + 4.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 10.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 14.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 7.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 11.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 15.0 * inc).xyz), 0.0, decodeFloat(texture(DiffuseSampler, start + 8.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 12.0 * inc).xyz), 0.0);

    mat4 ModeViewMat = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0,
                            0.0, 0.0, 0.0, 1.0);
    fogcol = vec4((texture(DiffuseSampler, start + 25.0 * inc)));
    skycol = vec4((texture(DiffuseSampler, start + 26.0 * inc)));
    GameTime = vec4((texture(DiffuseSampler, start + 27.0 * inc))).r;
    rain = vec4((texture(DiffuseSampler, start + 28.0 * inc)));
    near = PROJNEAR;
    far = ProjMat[3][2] * PROJNEAR / (ProjMat[3][2] + 2.0 * PROJNEAR);

    sunDir = normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + inc).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz),
                                                    1.0)).xyz);
    
    gbufferModelViewInverse = inverse(ProjMat * ModeViewMat);
    gbufferModelView = (ProjMat * ModeViewMat);
    gbufferProjection = ProjMat;
    gbufferProjectionInverse = inverse(ProjMat);
    aspectRatio = InSize.x / InSize.y;


}
