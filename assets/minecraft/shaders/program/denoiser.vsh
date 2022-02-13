#version 150

const float PROJNEAR = 0.05;
const float FPRECISION = 4000000.0;
const float EPSILON = 0.001;

in vec4 Position;

uniform mat4 ProjMat;
uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform sampler2D CurrentFrameDataSampler;
uniform sampler2D PreviousFrameDataSampler;
uniform sampler2D clouds;
uniform sampler2D prevclouds;

out vec2 texCoord;
out vec3 currChunkOffset;
out vec3 prevChunkOffset;
out vec2 oneTexel;
out mat4 gbufferProjection;
out mat4 gbufferProjectionInverse;
out mat4 gbufferModelView;
out mat4 gbufferModelViewInverse;
out mat4 wgbufferModelViewInverse;
out mat4 gbufferPreviousProjection;
out mat4 gbufferPreviousModelView;
out mat4 projInv;
out mat4 projInv2;
out vec3 rayDir;
out float overworld;
out float end;
out float near;
out float far;
out vec3 prevPosition;
out float exposure;
out float avgBrightness;
out float exposureF;
out float rodExposure;
out float avgL2;
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
float decodeFloat24(vec3 raw) {
uvec3 scaled = uvec3(raw * 255.0);
uint sign = scaled.r >> 7;
uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
return (- float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}
//Low discrepancy 2D sequence, integration error is as low as sobol but easier to compute : http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}
float luma(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}
void main() {
    vec4 outPos = ProjMat * vec4(Position.xy, 0, 1.0);
    gl_Position = vec4(outPos.xy, 0.2, 1.0);
    texCoord = Position.xy / OutSize;
    oneTexel = 1.0 / OutSize;



    vec2 start = getControl(0, OutSize);
    vec2 inc = vec2(2.0 / OutSize.x, 0.0);

    gbufferProjection = mat4(
        tan(decodeFloat(texture(CurrentFrameDataSampler, start + 3.0 * inc).xyz)), decodeFloat(texture(CurrentFrameDataSampler, start + 6.0 * inc).xyz), 0.0, 0.0,
        decodeFloat(texture(CurrentFrameDataSampler, start + 5.0 * inc).xyz), tan(decodeFloat(texture(CurrentFrameDataSampler, start + 4.0 * inc).xyz)), decodeFloat(texture(CurrentFrameDataSampler, start + 7.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 8.0 * inc).xyz),
        decodeFloat(texture(CurrentFrameDataSampler, start + 9.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 10.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 11.0 * inc).xyz),  decodeFloat(texture(CurrentFrameDataSampler, start + 12.0 * inc).xyz),
        decodeFloat(texture(CurrentFrameDataSampler, start + 13.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 14.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 15.0 * inc).xyz), 0.0
    );
    gbufferProjectionInverse = mat4(inverse(gbufferProjection));

    gbufferModelView = mat4(
        decodeFloat(texture(CurrentFrameDataSampler, start + 16.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 17.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 18.0 * inc).xyz), 0.0,
        decodeFloat(texture(CurrentFrameDataSampler, start + 19.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 20.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 21.0 * inc).xyz), 0.0,
        decodeFloat(texture(CurrentFrameDataSampler, start + 22.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 23.0 * inc).xyz), decodeFloat(texture(CurrentFrameDataSampler, start + 24.0 * inc).xyz), 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    gbufferModelViewInverse = mat4(inverse(gbufferModelView));

    gbufferPreviousProjection = mat4(
        tan(decodeFloat(texture(PreviousFrameDataSampler, start + 3.0 * inc).xyz)), decodeFloat(texture(PreviousFrameDataSampler, start + 6.0 * inc).xyz), 0.0, 0.0,
        decodeFloat(texture(PreviousFrameDataSampler, start + 5.0 * inc).xyz), tan(decodeFloat(texture(PreviousFrameDataSampler, start + 4.0 * inc).xyz)), decodeFloat(texture(PreviousFrameDataSampler, start + 7.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 8.0 * inc).xyz),
        decodeFloat(texture(PreviousFrameDataSampler, start + 9.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 10.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 11.0 * inc).xyz),  decodeFloat(texture(PreviousFrameDataSampler, start + 12.0 * inc).xyz),
        decodeFloat(texture(PreviousFrameDataSampler, start + 13.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 14.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 15.0 * inc).xyz), 0.0
    );

    gbufferPreviousModelView = mat4(
        decodeFloat(texture(PreviousFrameDataSampler, start + 16.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 17.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 18.0 * inc).xyz), 0.0,
        decodeFloat(texture(PreviousFrameDataSampler, start + 19.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 20.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 21.0 * inc).xyz), 0.0,
        decodeFloat(texture(PreviousFrameDataSampler, start + 22.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 23.0 * inc).xyz), decodeFloat(texture(PreviousFrameDataSampler, start + 24.0 * inc).xyz), 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    near = PROJNEAR;
    far = gbufferProjection[3][2] * PROJNEAR / (gbufferProjection[3][2] + 2.0 * PROJNEAR);
    
    wgbufferModelViewInverse = inverse(ProjMat * gbufferModelView);


    float cloudx = decodeFloat24((texture(clouds, start + 50.0 * inc).rgb));
    float cloudy = decodeFloat24((texture(clouds, start + 51.0 * inc).rgb));
    float cloudz = decodeFloat24((texture(clouds, start + 53.0 * inc).rgb));
    float cloudxprev = decodeFloat24((texture(prevclouds, start + 50.0 * inc).rgb));
    float cloudyprev = decodeFloat24((texture(prevclouds, start + 51.0 * inc).rgb));
    float cloudzprev = decodeFloat24((texture(prevclouds, start + 53.0 * inc).rgb));
    overworld = vec4((texture(CurrentFrameDataSampler, start + 28.0 * inc))).r;
    end = vec4((texture(CurrentFrameDataSampler, start + 29.0 * inc))).r;
    float fov = atan(1 / gbufferProjection[1][1]);
    currChunkOffset = vec3(0,cloudy*0.0,0);
    prevChunkOffset = vec3(0,cloudyprev*0.0,0);
    projInv = inverse(gbufferProjection * gbufferModelView);
    projInv2 = inverse(gbufferPreviousProjection * gbufferPreviousModelView);
    rayDir = (projInv * vec4(outPos.xy * (far - near), far + near, far - near)).xyz;
    prevPosition = currChunkOffset - prevChunkOffset;
    //prevPosition -= 16 * round(prevPosition / 16.0);

    vec2 oneTexel = 1 / ScreenSize;

	float avgLuma = 0.0;
	float m2 = 0.0;
	int n=100;
	vec2 clampedRes = max(1.0/oneTexel,vec2(1920.0,1080.));
	float avgExp = 0.0;
	float avgB = 0.0;
	vec2 resScale = vec2(1920.,1080.)/clampedRes*1.0;
	const int maxITexp = 50;
	float w = 0.0;
	for (int i = 0; i < maxITexp; i++){
			vec2 ij = R2_samples((1%2000)*maxITexp+i);
			vec2 tc = 0.5 + (ij-0.5) * 0.7;
			vec3 sp = texture2D(CurrentFrameDataSampler,tc/16. * resScale+vec2(0.375*resScale.x+4.5*oneTexel.x,.0)).rgb;
			avgExp += log(luma(sp));
			avgB += log(min(dot(sp,vec3(0.07,0.22,0.71)),8e-2));
	}

	avgExp = exp(avgExp/maxITexp);
	avgB = exp(avgB/maxITexp);

	avgBrightness = clamp(mix(avgExp,texelFetch(PreviousFrameDataSampler,ivec2(10,37),0).g,0.985),0.00003051757,65000.0);

	float L = max(avgBrightness,1e-8);
	float keyVal = 1.03-2.0/(log(L*4000/150.*8./3.0+1.0)/log(10.0)+2.0);
	float expFunc = 0.5+0.5*tanh(log(L));
	float targetExposure = 0.18/log2(L*2.5+1.045)*0.62;

	avgL2 = clamp(mix(avgB,texelFetch(PreviousFrameDataSampler,ivec2(10,37),0).b,0.985),0.00003051757,65000.0);
	float targetrodExposure = max(0.012/log2(avgL2+1.002)-0.1,0.0)*1.2;


	exposure=targetExposure*10.0;

}