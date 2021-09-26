#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D temporals3Sampler;
uniform sampler2D cloudsample;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TranslucentSampler;

uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float Time;
uniform mat4 ProjMat;

flat in vec3 ambientUp;
flat in vec3 ambientLeft;
flat in vec3 ambientRight;
flat in vec3 ambientB;
flat in vec3 ambientF;
flat in vec3 ambientDown;
flat in vec3 avgSky;

in vec2 texCoord;
in vec2 oneTexel;
in vec3 sunDir;
in vec4 fogcol;
in vec4 skycol;
in vec4 rain;
in mat4 gbufferModelViewInverse;
in mat4 gbufferModelView;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in float near;
in float far;
in float end;
in float overworld;
in float aspectRatio;

in float sunElevation;
in float rainStrength;
in vec3 sunVec;
in vec3 sunPosition;
in float skyIntensity;
in float skyIntensityNight;



out vec4 fragColor;

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define FUDGE 32.0




vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}



int inControl(vec2 screenCoord, float screenWidth) {
    if (screenCoord.y < 1.0) {
        float index = floor(screenWidth / 2.0) + THRESH / 2.0;
        index = (screenCoord.x - index) / 2.0;
        if (fract(index) < THRESH && index < NUMCONTROLS && index >= 0) {
            return int(index);
        }
    }
    return -1;
}

vec4 getNotControl(sampler2D inSampler, vec2 coords, bool inctrl) {
    if (inctrl) {
        return (texture(inSampler, coords - vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(0.0, oneTexel.y))) / 3.0;
    } else {
        return texture(inSampler, coords);
    }
}



float LinearizeDepth(float depth) 
{
    return (2.0 * near * far) / (far + near - depth * (far - near));    
}

float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}

vec4 backProject(vec4 vec) {
    vec4 tmp = gbufferModelViewInverse * vec;
    return tmp / tmp.w;
}




vec3 normVec (vec3 vec){
	return vec*inversesqrt(dot(vec,vec));
}

float ditherGradNoise() {
  return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}



float GetLinearDepth(float depth) {
   return (2.0 * near) / (far + near - depth * (far - near));
}

vec2 OffsetDist(float x, int s) {
	float n = fract(x * 1.414) * 3.1415;
    return vec2(cos(n), sin(n)) * x / s;
}

vec2 OffsetDist(float x) {
	float n = fract(x * 8.0) * 3.1415;
    return vec2(cos(n), sin(n)) * x;
}
//Dithering from Jodie
float Bayer2(vec2 a) {
    a = floor(a);
    return fract(dot(a, vec2(0.5, a.y * 0.75)));
}

#define Bayer4(a)   (Bayer2(  0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer8(a)   (Bayer4(  0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer16(a)  (Bayer8(  0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer32(a)  (Bayer16( 0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer64(a)  (Bayer32( 0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer128(a) (Bayer64( 0.5 * (a)) * 0.25 + Bayer2(a))
#define Bayer256(a) (Bayer128(0.5 * (a)) * 0.25 + Bayer2(a))


float AmbientOcclusion(sampler2D depth, vec2 coord, float dither) {
	float ao = 0.0;
	float far = far;
	float aspectRatio = ScreenSize.x/ScreenSize.y;
	if(overworld != 1.0 && end != 1.0) far = 1028;

		int samples = 6;


	float d = texture(depth, coord).r;
	if(d >= 1.0) return 1.0;
	float hand = float(d < 0.56);
	d = GetLinearDepth(d);
	
	float sampleDepth = 0.0, angle = 0.0, dist = 0.0;
	float fovScale = gbufferModelViewInverse[1][1] / 1.37;
	float distScale = max((far - near) * d + near, 6.0);
	vec2 scale = 0.35 * vec2(1.0 / aspectRatio, 1.0) * fovScale / distScale;
	scale *= vec2(0.5, 1.0);
	dither = fract(Time + dither);

	for(int i = 1; i <= samples; i++) {
		vec2 offset = OffsetDist(i + dither, samples) * scale;

		sampleDepth = GetLinearDepth(texture(depth, coord + offset).r);
		float sample = (far - near) * (d - sampleDepth) * 2.0;
		if (hand > 0.5) sample *= 1024.0;
		angle = clamp(0.5 - sample, 0.0, 1.0);
		dist = clamp(0.5 * sample - 1.0, 0.0, 1.0);

		sampleDepth = GetLinearDepth(texture(depth, coord - offset).r);
		sample = (far - near) * (d - sampleDepth) * 2.0;
		if (hand > 0.5) sample *= 1024.0;
		angle += clamp(0.5 - sample, 0.0, 1.0);
		dist += clamp(0.5 * sample - 1.0, 0.0, 1.0);
		
		ao += clamp(angle + dist, 0.0, 1.0);
	}
	ao /= samples;
	
	return ao;
}


float GGX (vec3 n, vec3 v, vec3 l, float r, float F0) {
  r*=r;r*=r;

  vec3 h = l + v;
  float hn = inversesqrt(dot(h, h));

  float dotLH = clamp(dot(h,l)*hn,0.,1.);
  float dotNH = clamp(dot(h,n)*hn,0.,1.);
  float dotNL = clamp(dot(n,l),0.,1.);
  float dotNHsq = dotNH*dotNH;

  float denom = dotNHsq * r - dotNHsq + 1.;
  float D = r / (3.141592653589793 * denom * denom);
  float F = F0 + (1. - F0) * exp2((-5.55473*dotLH-6.98316)*dotLH);
  float k2 = .25 * r;

  return dotNL * D * F / (dotLH*dotLH*(1.0-k2)+k2);
}


//GGX area light approximation from Horizon Zero Dawn
float GetNoHSquared(float radiusTan, float NoL, float NoV, float VoL) {
    float radiusCos = 1.0 / sqrt(1.0 + radiusTan * radiusTan);
    
    float RoL = 2.0 * NoL * NoV - VoL;
    if (RoL >= radiusCos)
        return 1.0;

    float rOverLengthT = radiusCos * radiusTan / sqrt(1.0 - RoL * RoL);
    float NoTr = rOverLengthT * (NoV - RoL * NoL);
    float VoTr = rOverLengthT * (2.0 * NoV * NoV - 1.0 - RoL * VoL);

    float triple = sqrt(clamp(1.0 - NoL * NoL - NoV * NoV - VoL * VoL + 2.0 * NoL * NoV * VoL, 0.0, 1.0));
    
    float NoBr = rOverLengthT * triple, VoBr = rOverLengthT * (2.0 * triple * NoV);
    float NoLVTr = NoL * radiusCos + NoV + NoTr, VoLVTr = VoL * radiusCos + 1.0 + VoTr;
    float p = NoBr * VoLVTr, q = NoLVTr * VoLVTr, s = VoBr * NoLVTr;    
    float xNum = q * (-0.5 * p + 0.25 * VoBr * NoLVTr);
    float xDenom = p * p + s * ((s - 2.0 * p)) + NoLVTr * ((NoL * radiusCos + NoV) * VoLVTr * VoLVTr + 
                   q * (-0.5 * (VoLVTr + VoL * radiusCos) - 0.5));
    float twoX1 = 2.0 * xNum / (xDenom * xDenom + xNum * xNum);
    float sinTheta = twoX1 * xDenom;
    float cosTheta = 1.0 - twoX1 * xNum;
    NoTr = cosTheta * NoTr + sinTheta * NoBr;
    VoTr = cosTheta * VoTr + sinTheta * VoBr;
    
    float newNoL = NoL * radiusCos + NoTr;
    float newVoL = VoL * radiusCos + VoTr;
    float NoH = NoV + newNoL;
    float HoH = 2.0 * newVoL + 2.0;
    return clamp(NoH * NoH / HoH, 0.0, 1.0);
}

float SchlickGGX(float NoL, float NoV, float roughness) {
    float k = roughness * 0.5;
        
    float smithL = 0.5 / (NoL * (1.0 - k) + k);
    float smithV = 0.5 / (NoV * (1.0 - k) + k);

    return smithL * smithV;
}

float GGX(vec3 normal, vec3 viewPos, vec3 lightVec, float smoothness, float f0, float sunSize) {
    float roughness = 1.0 - smoothness;
    if (roughness < 0.05) roughness = 0.05;
    float roughnessP = roughness;
    roughness *= roughness; roughness *= roughness;
    
    vec3 halfVec = normalize(lightVec - viewPos);

    float dotLH = clamp(dot(halfVec, lightVec), 0.0, 1.0);
    float dotNL = clamp(dot(normal,  lightVec), 0.0, 1.0);
    float dotNV = dot(normal, -viewPos);
    float dotNH = GetNoHSquared(sunSize, dotNL, dotNV, dot(-viewPos, lightVec));
    
    float denom = dotNH * roughness - dotNH + 1.0;
    float D = roughness / (3.141592653589793 * denom * denom);
    float F = exp2((-5.55473 * dotLH - 6.98316) * dotLH) * (1.0 - f0) + f0;
    float k2 = roughness * 0.5;

    float specular = max(dotNL * dotNL * D * F / (dotLH * dotLH * (1.0 - k2) + k2), 0.0);
    specular = max(specular, 0.0);
    specular = specular / (0.125 * specular + 1.0);

    float schlick = SchlickGGX(dotNL, dotNV, roughness);
    schlick = pow(schlick * 0.5, roughnessP);
    specular *= clamp(schlick, 0.0, 1.25);

    return specular ;
}



float w0(float a)
{
    return (1.0/6.0)*(a*(a*(-a + 3.0) - 3.0) + 1.0);
}

float w1(float a)
{
    return (1.0/6.0)*(a*a*(3.0*a - 6.0) + 4.0);
}

float w2(float a)
{
    return (1.0/6.0)*(a*(a*(-3.0*a + 3.0) + 3.0) + 1.0);
}

float w3(float a)
{
    return (1.0/6.0)*(a*a*a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0 + w1(a) / (w0(a) + w1(a));
}

float h1(float a)
{
    return 1.0 + w3(a) / (w2(a) + w3(a));
}
vec4 texture_bicubic(sampler2D tex, vec2 uv)
{
	vec4 texelSize = vec4(oneTexel,1.0/oneTexel);
	uv = uv*texelSize.zw;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );

    float g0x = g0(fuv.x);
    float g1x = g1(fuv.x);
    float h0x = h0(fuv.x);
    float h1x = h1(fuv.x);
    float h0y = h0(fuv.y);
    float h1y = h1(fuv.y);

	vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - 0.5) * texelSize.xy;
	vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - 0.5) * texelSize.xy;
	vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - 0.5) * texelSize.xy;
	vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - 0.5) * texelSize.xy;

    return g0(fuv.y) * (g0x * texture(tex, p0)  +
                        g1x * texture(tex, p1)) +
           g1(fuv.y) * (g0x * texture(tex, p2)  +
                        g1x * texture(tex, p3));
}


vec3 lumaBasedReinhardToneMapping(vec3 color)
{
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma / (1. + luma);
	color *= toneMappedLuma / luma;
	color = pow(color, vec3(1. / 2.2));
	return color;
}



vec3 getSkyColorLut(vec3 sVector, vec3 sunVec,float cosT,sampler2D lut) {
	const vec3 moonlight = vec3(0.8, 1.1, 1.4) * 0.06;

	float mCosT = clamp(cosT,0.0,1.);
	float cosY = dot(sunVec,sVector);
	float x = ((cosY*cosY)*(cosY*0.5*256.)+0.5*256.+18.+0.5)*oneTexel.x;
	float y = (mCosT*256.+1.0+0.5)*oneTexel.y;

	return texture(lut,vec2(x,y)).rgb;


}

vec3 drawSun(float cosY, float sunInt,vec3 nsunlight,vec3 inColor){
	return inColor+nsunlight/0.0008821203*pow(smoothstep(cos(0.0093084168595*3.2),cos(0.0093084168595*1.8),cosY),3.)*0.62;
}

// Return random noise in the range [0.0, 1.0], as a function of x.
float hash12(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
// Convert Noise2d() into a "star field" by stomping everthing below fThreshhold to zero.
float NoisyStarField( in vec2 vSamplePos, float fThreshhold )
{
    float StarVal = hash12( vSamplePos );
        StarVal = clamp(StarVal/(1.0 - fThreshhold) - fThreshhold/(1.0 - fThreshhold),0.0,1.0);

    return StarVal;
}

// Stabilize NoisyStarField() by only sampling at integer values.
float StableStarField( in vec2 vSamplePos, float fThreshhold )
{
    // Linear interpolation between four samples.
    // Note: This approach has some visual artifacts.
    // There must be a better way to "anti alias" the star field.
    float fractX = fract( vSamplePos.x );
    float fractY = fract( vSamplePos.y );
    vec2 floorSample = floor( vSamplePos );
    float v1 = NoisyStarField( floorSample, fThreshhold );
    float v2 = NoisyStarField( floorSample + vec2( 0.0, 1.0 ), fThreshhold );
    float v3 = NoisyStarField( floorSample + vec2( 1.0, 0.0 ), fThreshhold );
    float v4 = NoisyStarField( floorSample + vec2( 1.0, 1.0 ), fThreshhold );

    float StarVal =   v1 * ( 1.0 - fractX ) * ( 1.0 - fractY )
        			+ v2 * ( 1.0 - fractX ) * fractY
        			+ v3 * fractX * ( 1.0 - fractY )
        			+ v4 * fractX * fractY;
	return StarVal;
}

float stars(vec3 fragpos){

	float elevation = clamp(fragpos.y,0.,1.);
	vec2 uv = fragpos.xz/(1.+elevation);

	return StableStarField(uv*700.,0.999)*0.5*(0.3-0.3*0);
}

#define ffstep(x,y) clamp((y - x) * 1e35,0.0,1.0)

const float pi = 3.141592653589793238462643383279502884197169;









//Mie phase function
float phaseg(float x, float g){
    float gg = g * g;
    return (gg * -0.25 /3.14 + 0.25 /3.14) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5);
}



vec2 Nnoise(vec2 coord)
{
     float x = sin(coord.x * 100.0) * 0.1 + sin((coord.x * 200.0) + 3.0) * 0.05 + fract(cos((coord.x * 19.0) + 1.0) * 33.33) * 0.15;
     float y = sin(coord.y * 100.0) * 0.1 + sin((coord.y * 200.0) + 3.0) * 0.05 + fract(cos((coord.y * 19.0) + 1.0) * 33.33) * 0.25;
	 return vec2(x,y);
}




/////////////////////////////////
/*
uniform sampler2D FontSampler;  // ASCII 32x8 characters font texture unit


const float FXS = 0.02;         // font/screen resolution ratio
const float FYS = 0.02;         // font/screen resolution ratio

const int TEXT_BUFFER_LENGTH = 32;
int text[TEXT_BUFFER_LENGTH];
int textIndex;
vec4 colour;                    // color interface for printTextAt()

void floatToDigits(float x) {
    float y, a;
    const float base = 10.0;

    // Handle sign
    if (x < 0.0) { 
		text[textIndex] = '-'; textIndex++; x = -x; 
	} else { 
		text[textIndex] = '+'; textIndex++; 
	}

    // Get integer (x) and fractional (y) part of number
    y = x; 
    x = floor(x); 
    y -= x;

    // Handle integer part
    int i = textIndex;  // Start of integer part
    while (textIndex < TEXT_BUFFER_LENGTH) {
		// Get last digit, scale x down by 10 (or other base)
        a = x;
        x = floor(x / base);
        a -= base * x;
		// Add last digit to text array (results in reverse order)
        text[textIndex] = int(a) + '0'; textIndex++;
        if (x <= 0.0) break;
    }
    int j = textIndex - 1;  // End of integer part

	// In-place reverse integer digits
    while (i < j) {
        int chr = text[i]; 
		text[i] = text[j];
		text[j] = chr;
		i++; j--;
    }

	text[textIndex] = '.'; textIndex++;

    // Handle fractional part
    while (textIndex < TEXT_BUFFER_LENGTH) {
		// Get first digit, scale y up by 10 (or other base)
        y *= base;
        a = floor(y);
        y -= a;
		// Add first digit to text array
        text[textIndex] = int(a) + '0'; textIndex++;
        if (y <= 0.0) break;
    }

	// Terminante string
    text[textIndex] = 0;
}

void printTextAt(float x0, float y0) {
    // Fragment position **in char-units**, relative to x0, y0
    float x = texCoord.x/FXS; x -= x0;
    float y = 0.5*(1.0 - texCoord.y)/FYS; y -= y0;

    // Stop if not inside bbox
    if ((x < 0.0) || (x > float(textIndex)) || (y < 0.0) || (y > 1.0)) return;
    
    int i = int(x); // Char index of this fragment in text
    x -= float(i); // Fraction into this char

	// Grab pixel from correct char texture
    i = text[i];
    x += float(int(i - ((i/16)*16)));
    y += float(int(i/16));
    x /= 16.0; y /= 16.0; // Divide by character-sheet size (in chars)

	vec4 fontPixel = texture(FontSampler, vec2(x,y));

    colour = vec4(fontPixel.rgb*fontPixel.a + colour.rgb*colour.a*(1 - fontPixel.a), 1.0);
}

void clearTextBuffer() {
    for (int i = 0; i < TEXT_BUFFER_LENGTH; i++) {
        text[i] = 0;
    }
    textIndex = 0;
}

void c(int character) {
    // Adds character to text buffer, increments index for next character
    // Short name for convenience
    text[textIndex] = character; 
    textIndex++;
}


*/
#define BASE_FOG_AMOUNT 0.2 //[0.0 0.2 0.4 0.6 0.8 1.0 1.25 1.5 1.75 2.0 3.0 4.0 5.0 10.0 20.0 30.0 50.0 100.0 150.0 200.0]  Base fog amount amount (does not change the "cloudy" fog)
#define CLOUDY_FOG_AMOUNT 1.0 //[0.0 0.2 0.4 0.6 0.8 1.0 1.25 1.5 1.75 2.0 3.0 4.0 5.0]
#define FOG_TOD_MULTIPLIER 1.0 //[0.0 0.2 0.4 0.6 0.8 1.0 1.25 1.5 1.75 2.0 3.0 4.0 5.0] //Influence of time of day on fog amount
#define FOG_RAIN_MULTIPLIER 1.0 //[0.0 0.2 0.4 0.6 0.8 1.0 1.25 1.5 1.75 2.0 3.0 4.0 5.0] //Influence of rain on fog amount
///////////////////////////////////

float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 0.43015971 * Time);
}





vec3 LinearTosRGB(in vec3 color)
{
    vec3 x = color * 12.92f;
    vec3 y = 1.055f * pow(clamp(color,0.0,1.0), vec3(1.0f / 2.4f)) - 0.055f;

    vec3 clr = color;
    clr.r = color.r < 0.0031308f ? x.r : y.r;
    clr.g = color.g < 0.0031308f ? x.g : y.g;
    clr.b = color.b < 0.0031308f ? x.b : y.b;

    return clr;
}

vec3 reinhard(vec3 x){
x *= 1.66;
return x/(1.0+x);
}

vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}
vec3 rodSample(vec2 Xi)
{
	float r = sqrt(1.0f - Xi.x*Xi.y);
    float phi = 2 * 3.14159265359 * Xi.y;

    return normalize(vec3(cos(phi) * r, sin(phi) * r, Xi.x)).xzy;
}
#define SSAO_SAMPLES 6

#define ffstep(x,y) clamp((y - x) * 1e35,0.0,1.0)
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)



int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}


////////////////////////////////////////////

 float frameTimeCounter =  sunElevation*1000;


////////////////////////////////////////////


#define CLOUDS_QUALITY 0.5 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]

// this code is in the public domain
vec4 textureQuadratic( in sampler2D sam, in vec2 p )
{
    vec2 texSize = (textureSize(sam,0).xy); 
    

    //Roger/iq style
	p = p*texSize;
	vec2 i = floor(p);
	vec2 f = fract(p);
	p = i + f*0.5;
	p = p/texSize;
    f = f*f*(3.0-2.0*f); // optional for extra sweet
	vec2 w = 0.5/texSize;
	return mix(mix(texture(sam,p+vec2(0,0)),
                   texture(sam,p+vec2(w.x,0)),f.x),
               mix(texture(sam,p+vec2(0,w.y)),
                   texture(sam,p+vec2(w.x,w.y)),f.x), f.y);
    /*

    // paniq style (https://www.shadertoy.com/view/wtXXDl)
    vec2 f = fract(p*texSize);
    vec2 c = (f*(f-1.0)+0.5) / texSize;
    vec2 w0 = p - c;
    vec2 w1 = p + c;
    return (texture(sam, vec2(w0.x, w0.y))+
    	    texture(sam, vec2(w0.x, w1.y))+
    	    texture(sam, vec2(w1.x, w1.y))+
    	    texture(sam, vec2(w1.x, w0.y)))/4.0;
#endif    
*/
    
}
vec4 sample_biquadratic(sampler2D channel, vec2 res, vec2 uv) {
    vec2 q = fract(uv * res);
    vec2 c = (q*(q - 1.0) + 0.5) / res;
    vec2 w0 = uv - c;
    vec2 w1 = uv + c;
    vec4 s = texture(channel, vec2(w0.x, w0.y))
    	   + texture(channel, vec2(w0.x, w1.y))
    	   + texture(channel, vec2(w1.x, w1.y))
    	   + texture(channel, vec2(w1.x, w0.y));
	return s / 4.0;
}

// avoid hardware interpolation
vec4 sample_biquadratic_exact(sampler2D channel, vec2 uv) {
    vec2 res = (textureSize(channel,0).xy);
    vec2 q = fract(uv * res);
    ivec2 t = ivec2(uv * res);
    ivec3 e = ivec3(-1, 0, 1);
    vec4 s00 = texelFetch(channel, t + e.xx, 0);
    vec4 s01 = texelFetch(channel, t + e.xy, 0);
    vec4 s02 = texelFetch(channel, t + e.xz, 0);
    vec4 s12 = texelFetch(channel, t + e.yz, 0);
    vec4 s11 = texelFetch(channel, t + e.yy, 0);
    vec4 s10 = texelFetch(channel, t + e.yx, 0);
    vec4 s20 = texelFetch(channel, t + e.zx, 0);
    vec4 s21 = texelFetch(channel, t + e.zy, 0);
    vec4 s22 = texelFetch(channel, t + e.zz, 0);    
    vec2 q0 = (q+1.0)/2.0;
    vec2 q1 = q/2.0;	
    vec4 x0 = mix(mix(s00, s01, q0.y), mix(s01, s02, q1.y), q.y);
    vec4 x1 = mix(mix(s10, s11, q0.y), mix(s11, s12, q1.y), q.y);
    vec4 x2 = mix(mix(s20, s21, q0.y), mix(s21, s22, q1.y), q.y);    
	return mix(mix(x0, x1, q0.x), mix(x1, x2, q1.x), q.x);
}


vec4 textureBilinear(sampler2D tex, vec2 coord, const int res) {
    ivec2 texSize = textureSize(tex, 0)*res;
    vec2 texelSize = (1.0/vec2(texSize));
    vec4 p0q0 = texture(tex, coord);
    vec4 p1q0 = texture(tex, coord + vec2(texelSize.x, 0));

    vec4 p0q1 = texture(tex, coord + vec2(0, texelSize.y));
    vec4 p1q1 = texture(tex, coord + vec2(texelSize.x , texelSize.y));

    float a = fract(coord.x * texSize.x);

    vec4 pInterp_q0 = mix(p0q0, p1q0, a);
    vec4 pInterp_q1 = mix(p0q1, p1q1, a);

    float b = fract(coord.y*texSize.y);
    return mix(pInterp_q0, pInterp_q1, b);
}

vec3 reconstructPosition(in vec2 uv, in float z, in mat4  InvVP)
{
  float x = uv.x * 2.0f - 1.0f;
  float y = (1.0 - uv.y) * 2.0f - 1.0f;
  vec4 position_s = vec4(x, y, z, 1.0f);
  vec4 position_v =   InvVP*position_s;
  return position_v.xyz / position_v.w;
}




#define Dirt_Amount 0.01 



#define Dirt_Absorb_R 0.65 
#define Dirt_Absorb_G 0.85 
#define Dirt_Absorb_B 1.05

#define Water_Absorb_R 0.25422
#define Water_Absorb_G 0.03751
#define Water_Absorb_B 0.01150







#define Dirt_Mie_Phase 0.4  //Values close to 1 will create a strong peak around the sun and weak elsewhere, values close to 0 means uniform fog. [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 ]



void waterVolumetrics(inout vec3 inColor, vec3 rayStart, vec3 rayEnd, float estEyeDepth, float estSunDepth, float rayLength, float dither, vec3 waterCoefs, vec3 scatterCoef, vec3 ambient, vec3 lightSource, float VdotL, float sunElevation){
		int spCount = 6;
		//limit ray length at 32 blocks for performance and reducing integration error
		//you can't see above this anyway
		float maxZ = min(rayLength,32.0)/(1e-8+rayLength);
		rayLength *= maxZ;
		float dY = normalize(rayEnd).y * rayLength;
		vec3 absorbance = vec3(1.0);
		vec3 vL = vec3(0.0);
		float phase = phaseg(VdotL, Dirt_Mie_Phase);
		float expFactor = 11.0;
		for (int i=0;i<spCount;i++) {
			float d = (pow(expFactor, float(i+dither)/float(spCount))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);		// exponential step position (0-1)
			float dd = pow(expFactor, float(i+dither)/float(spCount)) * log(expFactor) / float(spCount)/(expFactor-1.0);	//step length (derivative)
			vec3 ambientMul = exp(-max(estEyeDepth - dY * d,0.0) * waterCoefs * 1.1);
			vec3 sunMul = exp(-max((estEyeDepth - dY * d) ,0.0)/abs(sunElevation) * waterCoefs);
			vec3 light = (0.75 * lightSource * phase * sunMul + ambientMul*ambient )*scatterCoef;
			vL += (light - light * exp(-waterCoefs * dd * rayLength)) / waterCoefs *absorbance;
			absorbance *= exp(-dd * rayLength * waterCoefs);
		}
		inColor += vL;
}

vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

vec4 pbr (vec2 in1,vec2 in2, float sssMin, float sssMax, float lightMin, float lightMax,float roughMin, float roughMax, float metalMin, float metalMax){

    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);

    vec4 alphatest = vec4(0.0);
    vec4 pbr = vec4(0.0);

    vec2 lmtrans = unpackUnorm2x4((texture(DiffuseSampler, texCoord).a));
    vec2 lmtrans3 = unpackUnorm2x4((texture(DiffuseSampler, texCoord+oneTexel.y).a));

    float maps1 = mix(in1.x,in2.x,res);
    float maps2 = mix(in2.x,in1.x,res);

    maps1 = map(maps1,  0, 1, 128, 255);
    if(maps1 == 128) maps1 = 0.0;
    maps2 = map(maps2,  0, 1, 0, 128);

    float maps = (maps1+maps2)/255;
    float expanded = int(maps * 255);

    if(expanded >=  18 && expanded <=  38)   alphatest.g = maps; // SSS
    float sss = map(alphatest.g*255,  18, 38,0,1);    

    if(expanded >=  39 && expanded <= 115)   alphatest.r = maps; // Emissives
    float emiss = map(alphatest.r*255, 39,115,0,1);    

    if(expanded >= 116 && expanded <= 208)   alphatest.b = maps; // Roughness
    float rough = map(alphatest.b*255,116,208,0,1);


    if(expanded >= 209 && expanded <= 251)   alphatest.a = maps; // Metals
    float metal = map(alphatest.a*255,209,251,0,1);
    if(rough < 0.001) rough = 0.1;

    pbr = vec4(emiss,sss,rough, metal);
    return pbr;    
}


void main() {
    float depth = texture(DiffuseDepthSampler, texCoord).r;
  	vec2 texCoord = texCoord; 
  	vec2 texCoord2 = texCoord; 

	if(overworld != 1.0 && end != 1.0){
    vec2 lmtrans2 = unpackUnorm2x4((texture(DiffuseSampler, texCoord).a));

    vec2 p_m = texCoord;
    vec2 p_d = p_m;
    p_d.xy -= Time * 0.1;
    vec2 dst_map_val = vec2(Nnoise(p_d.xy));
    vec2 dst_offset = dst_map_val.xy;

    dst_offset *= 2.0;

    dst_offset *= 0.01;
	
    //reduce effect towards Y top
	
    dst_offset *= (1. - p_m.t);	
    vec2 dist_tex_coord = p_m.st + (dst_offset*depth*0.2);

	vec2 coord = dist_tex_coord;

  	 texCoord = coord; 

	}


float mod2 = gl_FragCoord.x + gl_FragCoord.y;
float res = mod(mod2, 2.0f);





    depth = texture(DiffuseDepthSampler, texCoord).r;
    float deptht = texture(DiffuseDepthSampler, texCoord+oneTexel.y).r;
	vec3 vl = vec3(0.);

    bool inctrl = inControl(texCoord * OutSize, OutSize.x) > -1;
    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);

    float ao = AmbientOcclusion(TranslucentDepthSampler,texCoord,Bayer256(gl_FragCoord.xy)) ;


    vec3 sc = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb;
    vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));

    vec3 fragpos = backProject(vec4(scaledCoord, depth, 1.0)).xyz;
 


    vec2 lmtrans = unpackUnorm2x4((texture(DiffuseSampler, texCoord).a));
    vec2 lmtrans3 = unpackUnorm2x4((texture(DiffuseSampler, texCoord+oneTexel.y).a));

    
    #define sssMin 18
    #define sssMax 38
    #define lightMin 39
    #define lightMax 115
    #define roughMin 119
    #define roughMax 208
    #define metalMin 209
    #define metalMax 251

    vec4 pbr = pbr( lmtrans, lmtrans3, sssMin, sssMax, lightMin,  lightMax, roughMin, roughMax, metalMin, metalMax);

    float sssAmount = pbr.g;
    float ggxAmmount = pbr.b;
    float ggxAmmount2 = pbr.a;
    float light = pbr.r;

    if (depth > 1.0) light = 0;



    float lmx = 0;
    float lmy = 0;
    vec3 OutTexel = (texture(DiffuseSampler, texCoord).rgb);
//    vec3 OutTexel2 = (texture(DiffuseSampler, texCoord+oneTexel.y).rgb);

          lmy = mix(lmtrans.y,lmtrans3.y,res);
          lmx = mix(lmtrans3.y,lmtrans.y,res);
//          OutTexel = mix(OutTexel2,OutTexel,res);
          OutTexel = toLinear(OutTexel);
          if (deptht >= 1) lmx = 1;

//    float light = lmtrans.x;

 //         light = mix(lmtrans.x,lmtrans3.x,res);







   fragColor.rgb = OutTexel;	



	if(overworld == 1.0){





    float al = length(OutTexel);
  

    vec3 np3 = normVec(view);
    vec3 np2 = vec3(0,1,0);
  	vec3 suncol = sc;
    vec3 direct;
    vec3 ambient;
    direct = suncol;		
    

if (depth >=1){


    vec3 atmosphere = ((getSkyColorLut(view,sunPosition.xyz,view.y,temporals3Sampler)))  ;

 		if (np3.y > 0.){
			atmosphere += stars(np3)*clamp(1-rainStrength,0,1);
        	((atmosphere += pow((1.0 / (1.0 + dot(-sunPosition, np3))),0.3)*suncol.rgb*0.05)*0.001)*clamp(1-rainStrength,0,1);
            atmosphere += drawSun(dot(sunPosition,np3),0, suncol.rgb/150.,vec3(0.0))*clamp(1-rainStrength,0,1);
            atmosphere += drawSun(dot(-sunPosition,np3),0, atmosphere,vec3(0.0))*clamp(1-rainStrength,0,1);

            
		}
//    atmosphere = mix(skycol.rgb*luma(atmosphere*2), atmosphere,1-rainStrength);
//		atmosphere = mix(atmosphere*2.0,atmosphere,1-rainStrength);	
 
	vec4 cloud = textureQuadratic(cloudsample, texCoord*CLOUDS_QUALITY);





	atmosphere = atmosphere*cloud.a+(cloud.rgb*1.1);


    fragColor.rgb = reinhard(atmosphere) ;


}

    // only do lighting if not sky and sunDir exists
    if (LinearizeDepth(depth) < far - FUDGE && length(sunPosition) > 0.99) {
	fragColor.a = 1;
        // first calculate approximate surface normal using depth map

        
        float depth2 = getNotControl(TranslucentDepthSampler, texCoord + vec2(0.0, oneTexel.y), inctrl).r;
        float depth3 = getNotControl(TranslucentDepthSampler, texCoord + vec2(oneTexel.x, 0.0), inctrl).r;
        float depth4 = getNotControl(TranslucentDepthSampler, texCoord - vec2(0.0, oneTexel.y), inctrl).r;
        float depth5 = getNotControl(TranslucentDepthSampler, texCoord - vec2(oneTexel.x, 0.0), inctrl).r;


        vec3 p2 = backProject(vec4(scaledCoord + 1.0 * vec2(0.0, oneTexel.y), depth2, 1.0)).xyz;
        p2 = p2 - fragpos;
        vec3 p3 = backProject(vec4(scaledCoord + 1.0 * vec2(oneTexel.x, 0.0), depth3, 1.0)).xyz;
        p3 = p3 - fragpos;
        vec3 p4 = backProject(vec4(scaledCoord - 1.0 * vec2(0.0, oneTexel.y), depth4, 1.0)).xyz;
        p4 = p4 - fragpos;
        vec3 p5 = backProject(vec4(scaledCoord - 1.0 * vec2(oneTexel.x, 0.0), depth5, 1.0)).xyz;
        p5 = p5 - fragpos;
        vec3 normal = normalize(cross( p2,  p3)) 
                    + normalize(cross(-p4,  p3)) 
                    + normalize(cross( p2, -p5)) 
                    + normalize(cross(-p4, -p5));
        normal = normal == vec3(0.0) ? vec3(0.0, 1.0, 0.0) : normalize(-normal);
        

//        vec3 normal = normalize( (mix(OutTexel,OutTexel2,res)*res)*2-1 );



/*
 vec3 P =  reconstructPosition(texCoord, depth, gbufferProjectionInverse);
 
 normal = normalize(cross(dFdx(P), dFdy(P)));
*/

	vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.));

		vec3 ambientLight = ambientUp*mix(clamp(ambientCoefs.y,0.,1.), 0.166, sssAmount);
		ambientLight += ambientDown*mix(clamp(-ambientCoefs.y,0.,1.), 0.166, sssAmount);
		ambientLight += ambientRight*mix(clamp(ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientLeft*mix(clamp(-ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientB*mix(clamp(ambientCoefs.z,0.,1.), 0.166, sssAmount);
		ambientLight += ambientF*mix(clamp(-ambientCoefs.z,0.,1.), 0.166, sssAmount);
		ambientLight *= (1.0+rainStrength*0.2);
  
	


       bool t1 = sssAmount > 0.0;





	vec3 shading;
	float shadeDir = 0;
	float shadeDirS = 0;
	float shadeDirM = 0;
//    float sunSpec = ((GGX(normal,-normalize(view),  sunPosition, 0.75, 0.5)));

   // float sunSpec = ((GGX(normal,-normalize(view),  sunPosition, roughL, 0.05)));		
    float sunSpec = GGX(normal, normalize(view), sunPosition, ggxAmmount, 0.05, 0.01 * 1.0 + 0.06);
          if(ggxAmmount2 > 0.001)sunSpec = GGX(normal, normalize(view), sunPosition, ggxAmmount2, 0.8, 0.01 * 1.0 + 0.06)*10;
  //        sunSpec *= 10.0;
	vec3 SSS = vec3(0.0);
    float filt = (1-sssAmount)*0.95;
	vec3 extinction = 1.0 - OutTexel*0.85;    
	// Day
	if (skyIntensity > 0.00001)
	{

			SSS = exp(-filt*11.0*extinction) + 3.0*exp(-filt*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(view, sunPosition),0.85))*1.26*0.25*sssAmount,0.0,1.0);
			SSS *= scattering*3.0;
			SSS *= clamp(sqrt(lmx*2-1.5),0,1);

		shadeDirS = clamp(skyIntensity*10,0,1)*dot(normal, sunPosition);
//       	if(t1) shadeDirS = clamp(skyIntensity*10,0,1)*mix(max(phaseg(dot(view, sunPosition),0.45)*2, phaseg(dot(view, sunPosition),0.1))*3, shadeDirS, 0.35);
        if(t1) shadeDirS = clamp(skyIntensity*10,0,1)*luma(SSS);
	}
	// Night
	if (skyIntensityNight > 0.00001)
	{
		shadeDirM = (skyIntensityNight*dot(normal, -sunPosition))*0.01;
       	if(t1) shadeDirM = clamp(skyIntensityNight,0,1)*mix(max(phaseg(dot(view, -sunPosition),0.45), phaseg(dot(view, -sunPosition),0.1)), shadeDirS, 0.35);
	

	}
        shadeDir =  clamp(shadeDirS + shadeDirM,0,1);

		shading = ambientLight + mix(vec3(0.0), mix(direct*0.5,direct,sunSpec), shadeDir);
   
		ambientLight = mix(ambientLight*vec3(0.2,0.2,0.5)*2.0,ambientLight,1-rainStrength);	
		shading = mix(ambientLight,shading,1-rainStrength);	
 
        if(lmx == 1) lmx *= 0.75;

		shading = mix(vec3(1.0),shading,clamp((lmx)*5.0,0,1));
		shading = mix(shading,vec3(1.0),clamp((lmy*0.75),0,1));
        shading *= ao;
    
        vec3 dlight =   ( OutTexel * shading);
        if (light > 0.1)  dlight.rgb = OutTexel* pow(clamp(luma(OutTexel.rgb)-0.2,0.0,1.0)/0.65*0.65+0.35,2.0);
    	fragColor.rgb =  lumaBasedReinhardToneMapping(dlight);           		     
       if (light > 0.1)  fragColor.rgb *= clamp(vec3(2.0-shading*2),1.0,10.0);
    float isWater = 0;
    if (texture(TranslucentSampler, texCoord).a *255 ==200) isWater = 1;
   
   if (isWater == 1){


           float df = length(fragpos) ;
      float dirtAmount = Dirt_Amount;
      vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B)*fogcol.rgb;
      vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
      vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;
      fragColor.rgb *= clamp(exp(-df*totEpsilon),0.2,1.0);

    }



		fragColor.rgb = clamp(vec3(shading),0.01,1); 
    }




	}
	else{

	 fragColor.rgb =  mix(lumaBasedReinhardToneMapping(fragColor.rgb),fogcol.rgb*0.5,pow(depth,2048));
	}
	fragColor.a = texture(DiffuseSampler, texCoord2).a;
/*

	vec4 numToPrint = vec4(maps);

	// Define text to draw
    clearTextBuffer();
    c('R'); c('e'); c('d'); c(':'); c(' '); floatToDigits(numToPrint.r);
    printTextAt(1.0, 1.0);

    clearTextBuffer();
    c('G'); c('r'); c('e'); c('e'); c('n'); c(':'); c(' '); floatToDigits(numToPrint.g);
    printTextAt(1.0, 2.0);

    clearTextBuffer();
    c('B'); c('l'); c('u'); c('e'); c(':'); c(' '); floatToDigits(numToPrint.b);
    printTextAt(1.0, 3.0);

    clearTextBuffer();
    c('A'); c('l'); c('p'); c('h'); c('a'); c(':'); c(' '); floatToDigits(numToPrint.a);
    printTextAt(1.0, 4.0);

    fragColor += colour;
*/
}
