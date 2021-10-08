#version 150

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 InSize;
uniform float FOV;
uniform float Time;
uniform sampler2D temporals3Sampler;
uniform sampler2D DiffuseSampler;
uniform vec2 OutSize;
out mat4 gbufferModelViewInverse;
out mat4 gbufferModelView;
out mat4 gbufferProjection;
out mat4 gbufferProjectionInverse;
out float near;
out float far;
out vec3 sunDir;
out vec2 texCoord;
out vec2 oneTexel;
out float aspectRatio;
out float cosFOVrad;
out float tanFOVrad;
out mat4 gbPI;
out mat4 gbP;
out vec4 fogcol;
flat out vec3 ambientUp;
flat out vec3 ambientLeft;
flat out vec3 ambientRight;
flat out vec3 ambientB;
flat out vec3 ambientF;
flat out vec3 ambientDown;
flat out vec3 avgSky;
// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define FPRECISION 4000000.0
#define PROJNEAR 0.05

vec2 getControl(int index, vec2 screenSize) {
    return vec2(floor(screenSize.x / 2.0) + float(index) * 2.0 + 0.5, 0.5) / screenSize;
}
int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}

float decodeFloat(vec3 ivec) {
    return decodeInt(ivec) / FPRECISION;
}
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
vec3 skyLut(vec3 sVector, vec3 sunVec,float cosT,sampler2D lut) {
	const vec3 moonlight = vec3(0.8, 1.1, 1.4) * 0.06;

	float mCosT = clamp(cosT,0.0,1.);
	float cosY = dot(sunVec,sVector);
	float x = ((cosY*cosY)*(cosY*0.5*256.)+0.5*256.+18.+0.5)*oneTexel.x;
	float y = (mCosT*256.+1.0+0.5)*oneTexel.y;

	return texture(lut,vec2(x,y)).rgb;


}
void main(){
    vec4 outPos = ProjMat * vec4(Position.xy, 0.0, 1.0);
    gl_Position = vec4(outPos.xy, 0.2, 1.0);
    //simply decoding all the control data and constructing the sunDir, ProjMat, ModelViewMat

    vec2 start = getControl(0, OutSize);
    vec2 inc = vec2(2.0 / OutSize.x, 0.0);


    // ProjMat constructed assuming no translation or rotation matrices applied (aka no view bobbing).
    mat4 ProjMat = mat4(tan(decodeFloat(texture(DiffuseSampler, start + 3.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 6.0 * inc).xyz), 0.0, 0.0,
                        decodeFloat(texture(DiffuseSampler, start + 5.0 * inc).xyz), tan(decodeFloat(texture(DiffuseSampler, start + 4.0 * inc).xyz)), decodeFloat(texture(DiffuseSampler, start + 7.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 8.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 9.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 10.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 11.0 * inc).xyz),  decodeFloat(texture(DiffuseSampler, start + 12.0 * inc).xyz),
                        decodeFloat(texture(DiffuseSampler, start + 13.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 14.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 15.0 * inc).xyz), 0.0);
    gbufferModelView = mat4(
        decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0,
        decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0,
        decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    gbufferModelViewInverse = mat4(inverse(gbufferModelView));

    mat4 ModeViewMat = mat4(decodeFloat(texture(DiffuseSampler, start + 16.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 17.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 18.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 19.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 20.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 21.0 * inc).xyz), 0.0,
                            decodeFloat(texture(DiffuseSampler, start + 22.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 23.0 * inc).xyz), decodeFloat(texture(DiffuseSampler, start + 24.0 * inc).xyz), 0.0,
                            0.0, 0.0, 0.0, 1.0);
    sunDir = normalize((inverse(ModeViewMat) * vec4(decodeFloat(texture(DiffuseSampler, start).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + inc).xyz), 
                                                    decodeFloat(texture(DiffuseSampler, start + 2.0 * inc).xyz),
                                                    1.0)).xyz);

    fogcol = vec4((texture(DiffuseSampler, start + 25.0 * inc)));
    near = PROJNEAR;
    far = ProjMat[3][2] * PROJNEAR / (ProjMat[3][2] + 2.0 * PROJNEAR);


    gbufferProjection = (ProjMat);
    gbufferProjectionInverse = inverse(ProjMat);
	// normal = normalize(gl_NormalMatrix * normalize(vec3(0.0, 1.0, 0.0)));
    aspectRatio = InSize.x / InSize.y;
    texCoord = outPos.xy * 0.5 + 0.5;

    oneTexel = 1.0 / InSize;
    fogcol = vec4((texture(DiffuseSampler, start + 25.0 * inc)));

    float FOVrad = FOV / 360.0 * 3.1415926535;
    cosFOVrad = cos(FOVrad);
    tanFOVrad = tan(FOVrad);
    gbPI = mat4(2.0 * tanFOVrad * aspectRatio, 0.0,             0.0, 0.0,
                0.0,                           2.0 * tanFOVrad, 0.0, 0.0,
                0.0,                           0.0,             0.0, 0.0,
                -tanFOVrad * aspectRatio,     -tanFOVrad,       1.0, 1.0);

    gbP = mat4(1.0 / (2.0 * tanFOVrad * aspectRatio), 0.0,               0.0, 0.0,
               0.0,                             1.0 / (2.0 * tanFOVrad), 0.0, 0.0,
               0.5,                             0.5,                     1.0, 0.0,
               0.0,                             0.0,                     0.0, 1.0);


	avgSky = vec3(0.0);
    const int maxIT = 5;
	for (int i = 0; i < maxIT; i++) {
			vec2 ij = R2_samples((int(Time)%1000)*maxIT+i);
			vec3 pos = normalize(rodSample(ij));
			vec3 samplee = 2.2*skyLut(pos.xyz,sunDir,pos.y,temporals3Sampler)/maxIT;
			avgSky += samplee/2.2;
	}

}
