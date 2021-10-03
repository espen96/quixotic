#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D temporals3Sampler;
uniform sampler2D cloudsample;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D PreviousFrameSampler;
uniform sampler2D prevsky;

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

in float cosFOVrad;
in float tanFOVrad;
in mat4 gbPI;
in mat4 gbP;

out vec4 fragColor;


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


// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FUDGE 32.0

#define Dirt_Amount 0.01 


#define Dirt_Mie_Phase 0.4  //Values close to 1 will create a strong peak around the sun and weak elsewhere, values close to 0 means uniform fog. [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 ]


#define Dirt_Absorb_R 0.65 
#define Dirt_Absorb_G 0.85 
#define Dirt_Absorb_B 1.05

#define Water_Absorb_R 0.25422
#define Water_Absorb_G 0.03751
#define Water_Absorb_B 0.01150

#define BASE_FOG_AMOUNT 0.2 //[0.0 0.2 0.4 0.6 0.8 1.0 1.25 1.5 1.75 2.0 3.0 4.0 5.0 10.0 20.0 30.0 50.0 100.0 150.0 200.0]  Base fog amount amount (does not change the "cloudy" fog)
#define CLOUDY_FOG_AMOUNT 1.0 //[0.0 0.2 0.4 0.6 0.8 1.0 1.25 1.5 1.75 2.0 3.0 4.0 5.0]
#define FOG_TOD_MULTIPLIER 1.0 //[0.0 0.2 0.4 0.6 0.8 1.0 1.25 1.5 1.75 2.0 3.0 4.0 5.0] //Influence of time of day on fog amount
#define FOG_RAIN_MULTIPLIER 1.0 //[0.0 0.2 0.4 0.6 0.8 1.0 1.25 1.5 1.75 2.0 3.0 4.0 5.0] //Influence of rain on fog amount

#define SSAO_SAMPLES 6

#define NORMDEPTHTOLERANCE 1.0



#define SSR_SAMPLES 64
#define SSR_MAXREFINESAMPLES 1
#define SSR_STEPREFINE 0.1
#define SSR_STEPINCREASE 0.1
#define SSR_IGNORETHRESH 0.0

#define NORMAL_SCATTER 0.006



#define CLOUDS_QUALITY 0.5 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]


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



float GetLinearDepth(float depth) {
   return (2.0 * near) / (far + near - depth * (far - near));
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





vec2 OffsetDist(float x, int s) {
	float n = fract(x * 1.414) * 3.1415;
    return vec2(cos(n), sin(n)) * x / s;
}


float AmbientOcclusion(sampler2D depth, vec2 coord, float dither) {
	float ao = 0.0;
	float far = far;
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

///////////////////////////////////



vec3 reinhard(vec3 x){
x *= 1.66;
return x/(1.0+x);
}

vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}



int decodeInt(vec3 ivec) {
    ivec *= 255.0;
    int s = ivec.b >= 128.0 ? -1 : 1;
    return s * (int(ivec.r) + int(ivec.g) * 256 + (int(ivec.b) - 64 + s * 64) * 256 * 256);
}


////////////////////////////////////////////

 float frameTimeCounter =  sunElevation*1000;


////////////////////////////////////////////



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




vec3 reconstructPosition(in vec2 uv, in float z, in mat4  InvVP)
{
  float x = uv.x * 2.0f - 1.0f;
  float y = (1.0 - uv.y) * 2.0f - 1.0f;
  vec4 position_s = vec4(x, y, z, 1.0f);
  vec4 position_v =   InvVP*position_s;
  return position_v.xyz / position_v.w;
}




//////////////////////////////////////////////////////////////////////////////////////////
vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}
    #define sssMin 22
    #define sssMax 47
    #define lightMin 48
    #define lightMax 72
    #define roughMin 73
    #define roughMax 157
    #define metalMin 158
    #define metalMax 251

vec4 pbr (vec2 in1,vec2 in2){

    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);

    vec4 alphatest = vec4(0.0);
    vec4 pbr = vec4(0.0);


    float maps1 = mix(in1.x,in2.x,res);
    float maps2 = mix(in2.x,in1.x,res);

    maps1 = map(maps1,  0, 1, 128, 255);
    if(maps1 == 128) maps1 = 0.0;
    maps2 = map(maps2,  0, 1, 0, 128);

    float maps = (maps1+maps2)/255;
    float expanded = int(maps * 255);

    if(expanded >=  sssMin && expanded <=  sssMax)   alphatest.g = maps; // SSS
    float sss = map(alphatest.g*255,  sssMin, sssMax,0,1);    

    if(expanded >=  lightMin && expanded <= lightMax)   alphatest.r = maps; // Emissives
    float emiss = map(alphatest.r*255, lightMin,lightMax,0,1);    

    if(expanded >= roughMin && expanded <= roughMax)   alphatest.b = maps; // Roughness
    float rough = map(alphatest.b*255,roughMin,roughMax,0,1);


    if(expanded >= metalMin && expanded <= metalMax)   alphatest.a = maps; // Metals
    float metal = map(alphatest.a*255,metalMin,metalMax,0,1);
    if(rough < 0.001) rough = 0.1;

    pbr = vec4(emiss,sss,rough, metal);
 //   if(expanded > 170) pbr *=0;
    return pbr;    
}

/////////////////////////////////////////////////////////////////////////

float square(float x){
  return x*x;
}

float g(float NdotL, float roughness)
{
    float alpha = square(max(roughness, 0.02));
    return 2.0 * NdotL / (NdotL + sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotL)));
}

float gSimple(float dp, float roughness){
  float k = roughness + 1;
  k *= k/8.0;
  return dp / (dp * (1.0-k) + k);
}

vec3 GGX2(vec3 n, vec3 v, vec3 l, float r, vec3 F0) {
  float alpha = square(r);

  vec3 h = normalize(l + v);

  float dotLH = clamp(dot(h,l),0.,1.);
  float dotNH = clamp(dot(h,n),0.,1.);
  float dotNL = clamp(dot(n,l),0.,1.);
  float dotNV = clamp(dot(n,v),0.,1.);
  float dotVH = clamp(dot(h,v),0.,1.);


  float D = alpha / (3.141592653589793*square(square(dotNH) * (alpha - 1.0) + 1.0));
  float G = gSimple(dotNV, r) * gSimple(dotNL, r);
  vec3 F = F0 + (1. - F0) * exp2((-5.55473*dotVH-6.98316)*dotVH);

  return dotNL * F * (G * D / (4 * dotNV * dotNL + 1e-7));
}

vec3 GGX (vec3 n, vec3 v, vec3 l, float r, vec3 F0) {
  r*=r;r*=r;

  vec3 h = l + v;
  float hn = inversesqrt(dot(h, h));

  float dotLH = clamp(dot(h,l)*hn,0.,1.);
  float dotNH = clamp(dot(h,n)*hn,0.,1.);
  float dotNL = clamp(dot(n,l),0.,1.);
  float dotNHsq = dotNH*dotNH;

  float denom = dotNHsq * r - dotNHsq + 1.;
  float D = r / (3.141592653589793 * denom * denom);
  vec3 F = F0 + (1. - F0) * exp2((-5.55473*dotLH-6.98316)*dotLH);
  float k2 = .25 * r;

  return dotNL * D * F / (dotLH*dotLH*(1.0-k2)+k2);
}
void frisvad(in vec3 n, out vec3 f, out vec3 r){
    if(n.z < -0.999999) {
        f = vec3(0.,-1,0);
        r = vec3(-1, 0, 0);
    } else {
    	float a = 1./(1.+n.z);
    	float b = -n.x*n.y*a;
    	f = vec3(1. - n.x*n.x*a, b, -n.x);
    	r = vec3(b, 1. - n.y*n.y*a , -n.y);
    }
}
mat3 CoordBase(vec3 n){
	vec3 x,y;
    frisvad(n,x,y);
    return mat3(x,y,n);
}

vec3 MetalCol(float f0){
    int metalidx = int(f0 * 255.0);

    if (metalidx == 230) return vec3(0.24867, 0.22965, 0.21366); //iron
    if (metalidx == 231) return vec3(0.88140, 0.57256, 0.11450); //gold
    if (metalidx == 232) return vec3(0.81715, 0.82021, 0.83177); //aluminium
    if (metalidx == 233) return vec3(0.27446, 0.27330, 0.27357); //chrome
    if (metalidx == 234) return vec3(0.84430, 0.48677, 0.22164); //copper
    if (metalidx == 235) return vec3(0.36501, 0.35675, 0.37653); //lead
    if (metalidx == 236) return vec3(0.42648, 0.37772, 0.31138); //platinum
    if (metalidx == 237) return vec3(0.91830, 0.89219, 0.83662); //silver
    return vec3(1.0);
}									
	
vec3 worldToView(vec3 worldPos) {

    vec4 pos = vec4(worldPos, 0.0);
    pos = inverse(gbufferModelViewInverse) * pos;

    return pos.xyz;
}


vec3 sampleGGXVNDF(vec3 V_, float alpha_x, float alpha_y, float U1, float U2){
	// stretch view
	vec3 V = normalize(vec3(alpha_x * V_.x, alpha_y * V_.y, V_.z));
	// orthonormal basis
	vec3 T1 = (V.z < 0.9999) ? normalize(cross(V, vec3(0,0,1))) : vec3(1,0,0);
	vec3 T2 = cross(T1, V);
	// sample point with polar coordinates (r, phi)
	float a = 1.0 / (1.0 + V.z);
	float r = sqrt(U1);
	float phi = (U2<a) ? U2/a * 3.141592653589793 : 3.141592653589793 + (U2-a)/(1.0-a) * 3.141592653589793;
	float P1 = r*cos(phi);
	float P2 = r*sin(phi)*((U2<a) ? 1.0 : V.z);
	// compute normal
	vec3 N = P1*T1 + P2*T2 + sqrt(max(0.0, 1.0 - P1*P1 - P2*P2))*V;
	// unstretch
	N = normalize(vec3(alpha_x*N.x, alpha_y*N.y, max(0.0, N.z)));
	return N;
}



vec3 SSR(vec3 fragpos, float fragdepth, vec3 surfacenorm, vec4 skycol) {
    vec3 rayStart   = fragpos.xyz;
    vec3 rayDir     = surfacenorm;
//        vec3 rayDir     = reflect(normalize(fragpos.xyz), surfacenorm);
    vec3 rayStep    = 0.5 * rayDir;
    vec3 rayPos     = rayStart + rayStep;
    vec3 rayRefine  = rayStep;

    int refine  = 0;
    vec3 pos    = vec3(0.0);
    float dtmp  = 0.0;

    for (int i = 0; i < SSR_SAMPLES; i += 1) {
        pos = (gbP * vec4(rayPos.xyz, 1.0)).xyz;
        pos.xy /= rayPos.z;
		if (pos.x < -0.05 || pos.x > 1.05 || pos.y < -0.05 || pos.y > 1.05) break;
        dtmp = LinearizeDepth(texture(DiffuseDepthSampler, pos.xy).r);
        float dist = abs(rayPos.z - dtmp);

        if (dtmp + SSR_IGNORETHRESH > fragdepth && dist < length(rayStep) * pow(length(rayRefine), 0.11) * 2.0) {
            refine++;
            if (refine >= SSR_MAXREFINESAMPLES)	break;
            rayRefine  -= rayStep;
            rayStep    *= SSR_STEPREFINE;
        }

        rayStep        *= SSR_STEPINCREASE;
        rayRefine      += rayStep;
        rayPos          = rayStart+rayRefine;

    }


    vec4 candidate = vec4(0.0);
    if (fragdepth < dtmp + SSR_IGNORETHRESH && pos.y <= 1.0) {
        vec3 colortmp = texture(DiffuseSampler, pos.xy).rgb;



        candidate = mix(vec4(colortmp, 1.0), skycol, float(dtmp + SSR_IGNORETHRESH < 1.0) * clamp(pos.z * 1.1, 0.0, 1.0));
    }
    
    candidate = mix(candidate, skycol, pos.y );

    return candidate.xyz;
}
vec3 reinhard_jodie(vec3 v)
{
    float l = luma(v);
    vec3 tv = v / (1.0f + v);
    tv = mix(v / (1.0f + l), tv, tv);
    return 	pow(tv, vec3(1. / 2.2));
}



void main() {
  	vec2 texCoord = texCoord; 
  	vec2 texCoord2 = texCoord; 
    vec2 lmtrans = unpackUnorm2x4((texture(DiffuseSampler, texCoord).a));
    float deptha = texture(DiffuseDepthSampler, texCoord).r;
    if(deptha >= 1) lmtrans = vec2(0.0); 
    vec2 lmtrans2 = unpackUnorm2x4((texture(DiffuseSampler, texCoord-vec2(0,oneTexel.y)).a));
    float depthb = texture(DiffuseDepthSampler, texCoord-vec2(0,oneTexel.y)).r;
    lmtrans2 *= 1-(depthb -deptha);

    vec2 lmtrans3 = unpackUnorm2x4((texture(DiffuseSampler, texCoord+vec2(0,oneTexel.y)).a));
    float depthc = texture(DiffuseDepthSampler, texCoord+vec2(0,oneTexel.y)).r;
    lmtrans3 *= 1-(depthc -deptha);

    vec2 lmtrans4 = unpackUnorm2x4((texture(DiffuseSampler, texCoord+vec2(oneTexel.x,0)).a));
    float depthd = texture(DiffuseDepthSampler, texCoord+vec2(oneTexel.x,0)).r;
    lmtrans4 *= 1-(depthd -deptha);

    vec2 lmtrans5 = unpackUnorm2x4((texture(DiffuseSampler, texCoord-vec2(oneTexel.x,0)).a));
    float depthe = texture(DiffuseDepthSampler, texCoord-vec2(oneTexel.x,0)).r;
    lmtrans5 *= 1-(depthe -deptha);
    
    float depthtest = (deptha+depthb+depthc+depthd+depthe)/5;
    vec4 pbr = pbr( lmtrans,  unpackUnorm2x4((texture(DiffuseSampler, texCoord+vec2(oneTexel.y)).a)) );
    if( (depthtest-deptha)*1000 >0.1) pbr =vec4(0.0);
    float sssAmount = pbr.g;
    float ggxAmmount = pbr.b;
    float ggxAmmount2 = pbr.a;
    float light = pbr.r;
    float depth = deptha;
    if (depth > 1.0) light = 0;

	if(overworld != 1.0 && end != 1.0){

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



    vec3 OutTexel = (texture(DiffuseSampler, texCoord).rgb);
         OutTexel = toLinear(OutTexel);    


   fragColor.rgb = OutTexel;	


    float ao = AmbientOcclusion(DiffuseDepthSampler,texCoord,Bayer256(gl_FragCoord.xy)) ;
if(overworld == 1.0){
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);





	vec3 vl = vec3(0.);

    bool inctrl = inControl(texCoord * OutSize, OutSize.x) > -1;
    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);



    vec3 sc = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb;
    vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));

    vec3 fragpos = backProject(vec4(scaledCoord, depth, 1.0)).xyz;
 







    float lmx = 0;
    float lmy = 0;
    float postlight = 1;


        lmy = mix(lmtrans.y,(lmtrans2.y+lmtrans3.y+lmtrans4.y+lmtrans5.y)/4,res);
        lmx = mix((lmtrans2.y+lmtrans3.y+lmtrans4.y+lmtrans5.y)/4,lmtrans.y,res);

        if(lmx == 1) {
            lmx *= 0.75;
            postlight = 0.0;
            
        }

    vec3 lightmap = texture2D(temporals3Sampler,vec2(lmy,lmx)*(oneTexel*17)).xyz;

    if(postlight == 1)    OutTexel *= lightmap;
//     if (deptht >= 1 || depthy >= 1) discard;   







  

    vec3 np3 = normVec(view);
    vec3 np2 = vec3(0,1,0);
  	vec3 suncol = sc;
    vec3 direct;
    vec3 ambient;
    direct = suncol;		
    

 if (depthtest >= 1.0 || luma(OutTexel) == 0){


    vec3 atmosphere = ((getSkyColorLut(view,sunPosition.xyz,view.y,temporals3Sampler)))  ;

 		if (np3.y > 0.){
			atmosphere += stars(np3)*clamp(1-rainStrength,0,1);
        	((atmosphere += pow((1.0 / (1.0 + dot(-sunPosition, np3))),0.3)*suncol.rgb*0.05)*0.001)*clamp(1-rainStrength,0,1);
            atmosphere += drawSun(dot(sunPosition,np3),0, suncol.rgb/150.,vec3(0.0))*clamp(1-rainStrength,0,1);
            atmosphere += drawSun(dot(-sunPosition,np3),0, atmosphere,vec3(0.0))*clamp(1-rainStrength,0,1);

            
		}

	vec4 cloud = textureQuadratic(cloudsample, texCoord*CLOUDS_QUALITY);





	atmosphere = atmosphere*cloud.a+(cloud.rgb*1.1);


    fragColor.rgb = reinhard(atmosphere) ;

    return;
 }

    // only do lighting if not sky and sunDir exists
    if (LinearizeDepth(depth) < far - FUDGE && length(sunPosition) > 0.99) {
	fragColor.a = 1;

        // first calculate approximate surface normal using depth map

        
        float depth2 = getNotControl(TranslucentDepthSampler, texCoord + vec2(0.0, oneTexel.y), inctrl).r;
        float depth3 = getNotControl(TranslucentDepthSampler, texCoord + vec2(oneTexel.x, 0.0), inctrl).r;
        float depth4 = getNotControl(TranslucentDepthSampler, texCoord - vec2(0.0, oneTexel.y), inctrl).r;
        float depth5 = getNotControl(TranslucentDepthSampler, texCoord - vec2(oneTexel.x, 0.0), inctrl).r;


        vec3 p2 = backProject(vec4(scaledCoord + 2.0 * vec2(0.0, oneTexel.y), depth2, 1.0)).xyz;
        p2 = p2 - fragpos;
        vec3 p3 = backProject(vec4(scaledCoord + 2.0 * vec2(oneTexel.x, 0.0), depth3, 1.0)).xyz;
        p3 = p3 - fragpos;
        vec3 p4 = backProject(vec4(scaledCoord - 2.0 * vec2(0.0, oneTexel.y), depth4, 1.0)).xyz;
        p4 = p4 - fragpos;
        vec3 p5 = backProject(vec4(scaledCoord - 2.0 * vec2(oneTexel.x, 0.0), depth5, 1.0)).xyz;
        p5 = p5 - fragpos;
        vec3 normal = normalize(cross( p2,  p3)) 
                    + normalize(cross(-p4,  p3)) 
                    + normalize(cross( p2, -p5)) 
                    + normalize(cross(-p4, -p5));
        normal = normal == vec3(0.0) ? vec3(0.0, 1.0, 0.0) : normalize(-normal);


	    vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.));

		vec3 ambientLight = ambientUp*mix(clamp(ambientCoefs.y,0.,1.), 0.166, sssAmount);
		ambientLight += ambientDown*mix(clamp(-ambientCoefs.y,0.,1.), 0.166, sssAmount);
		ambientLight += ambientRight*mix(clamp(ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientLeft*mix(clamp(-ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientB*mix(clamp(ambientCoefs.z,0.,1.), 0.166, sssAmount);
		ambientLight += ambientF*mix(clamp(-ambientCoefs.z,0.,1.), 0.166, sssAmount);
		ambientLight *= (1.0+rainStrength*0.2);
  
	


       bool t1 = sssAmount > 0.0;
    vec2 poissonDisk[64];
    poissonDisk[0] = vec2(-0.613392, 0.617481);
    poissonDisk[1] = vec2(0.170019, -0.040254);
    poissonDisk[2] = vec2(-0.299417, 0.791925);
    poissonDisk[3] = vec2(0.645680, 0.493210);
    poissonDisk[4] = vec2(-0.651784, 0.717887);
    poissonDisk[5] = vec2(0.421003, 0.027070);
    poissonDisk[6] = vec2(-0.817194, -0.271096);
    poissonDisk[7] = vec2(-0.705374, -0.668203);
    poissonDisk[8] = vec2(0.977050, -0.108615);
    poissonDisk[9] = vec2(0.063326, 0.142369);
    poissonDisk[10] = vec2(0.203528, 0.214331);
    poissonDisk[11] = vec2(-0.667531, 0.326090);
    poissonDisk[12] = vec2(-0.098422, -0.295755);
    poissonDisk[13] = vec2(-0.885922, 0.215369);
    poissonDisk[14] = vec2(0.566637, 0.605213);
    poissonDisk[15] = vec2(0.039766, -0.396100);
    poissonDisk[16] = vec2(0.751946, 0.453352);
    poissonDisk[17] = vec2(0.078707, -0.715323);
    poissonDisk[18] = vec2(-0.075838, -0.529344);
    poissonDisk[19] = vec2(0.724479, -0.580798);
    poissonDisk[20] = vec2(0.222999, -0.215125);
    poissonDisk[21] = vec2(-0.467574, -0.405438);
    poissonDisk[22] = vec2(-0.248268, -0.814753);
    poissonDisk[23] = vec2(0.354411, -0.887570);
    poissonDisk[24] = vec2(0.175817, 0.382366);
    poissonDisk[25] = vec2(0.487472, -0.063082);
    poissonDisk[26] = vec2(-0.084078, 0.898312);
    poissonDisk[27] = vec2(0.488876, -0.783441);
    poissonDisk[28] = vec2(0.470016, 0.217933);
    poissonDisk[29] = vec2(-0.696890, -0.549791);
    poissonDisk[30] = vec2(-0.149693, 0.605762);
    poissonDisk[31] = vec2(0.034211, 0.979980);
    poissonDisk[32] = vec2(0.503098, -0.308878);
    poissonDisk[33] = vec2(-0.016205, -0.872921);
    poissonDisk[34] = vec2(0.385784, -0.393902);
    poissonDisk[35] = vec2(-0.146886, -0.859249);
    poissonDisk[36] = vec2(0.643361, 0.164098);
    poissonDisk[37] = vec2(0.634388, -0.049471);
    poissonDisk[38] = vec2(-0.688894, 0.007843);
    poissonDisk[39] = vec2(0.464034, -0.188818);
    poissonDisk[40] = vec2(-0.440840, 0.137486);
    poissonDisk[41] = vec2(0.364483, 0.511704);
    poissonDisk[42] = vec2(0.034028, 0.325968);
    poissonDisk[43] = vec2(0.099094, -0.308023);
    poissonDisk[44] = vec2(0.693960, -0.366253);
    poissonDisk[45] = vec2(0.678884, -0.204688);
    poissonDisk[46] = vec2(0.001801, 0.780328);
    poissonDisk[47] = vec2(0.145177, -0.898984);
    poissonDisk[48] = vec2(0.062655, -0.611866);
    poissonDisk[49] = vec2(0.315226, -0.604297);
    poissonDisk[50] = vec2(-0.780145, 0.486251);
    poissonDisk[51] = vec2(-0.371868, 0.882138);
    poissonDisk[52] = vec2(0.200476, 0.494430);
    poissonDisk[53] = vec2(-0.494552, -0.711051);
    poissonDisk[54] = vec2(0.612476, 0.705252);
    poissonDisk[55] = vec2(-0.578845, -0.768792);
    poissonDisk[56] = vec2(-0.772454, -0.090976);
    poissonDisk[57] = vec2(0.504440, 0.372295);
    poissonDisk[58] = vec2(0.155736, 0.065157);
    poissonDisk[59] = vec2(0.391522, 0.849605);
    poissonDisk[60] = vec2(-0.620106, -0.328104);
    poissonDisk[61] = vec2(0.789239, -0.419965);
    poissonDisk[62] = vec2(-0.545396, 0.538133);
    poissonDisk[63] = vec2(-0.178564, -0.596057);




	vec3 shading;
	float shadeDir = 0;
	float shadeDirS = 0;
	float shadeDirM = 0;

			vec3 f0 = vec3(0.04);
            if(ggxAmmount2 > 0.001) {f0 = vec3(0.8);
            ggxAmmount = ggxAmmount2;}
  //          if(ggxAmmount*255 <12) ggxAmmount = length(OutTexel); 
     //       if(ggxAmmount<0.05) ggxAmmount += (length(OutTexel));
            float sunSpec = ((GGX(normal,-normalize(view),  sunPosition, 1-ggxAmmount, f0.x)));		


   vec3 normal2 = normalize(worldToView(normal) );
			float roughness = 1-ggxAmmount;
			vec3 specTerm = GGX2(normal, -normalize(view),  sunPosition, roughness+0.05*0.95, f0);
            specTerm += vec3(sunSpec)*0.5;
            normalize(specTerm);
			vec3 indirectSpecular = vec3(0.0);
         
			const int nSpecularSamples = 6;
			// Energy conservation between diffuse and specular
			vec3 fresnelDiffuse = vec3(0.0);
			mat3 basis = CoordBase(normal2);
			vec3 normSpaceView = -np3*basis;
			vec3 rayContrib = vec3(0.0);
			vec3 reflection = vec3(0.0);
        if(f0.x >0.5){  

            OutTexel *= 0.75;
            float wdepth = texture(TranslucentDepthSampler, texCoord).r;

            float ldepth = LinearizeDepth(wdepth);
            vec3 fragpos3 = (gbPI * vec4(texCoord, ldepth, 1.0)).xyz;
            fragpos3 *= ldepth;
			for (int i = 0; i < nSpecularSamples; i++){
				// Generate ray
				int seed = int(Time*1000)*nSpecularSamples + i;
				vec2 ij = fract(R2_samples(seed) + Bayer256(gl_FragCoord.xy));
				vec3 H = sampleGGXVNDF(normSpaceView, roughness, roughness, ij.x, ij.y);
				vec3 Ln = reflect(-normSpaceView, H);
				vec3 L = basis * Ln;
				// Ray contribution
				float g1 = g(clamp(dot(normal2, L),0.0,1.0), roughness);
				vec3 F = f0 + (1.0 - f0) * pow(clamp(1.0 + dot(-Ln, H),0.0,1.0), 5.0);

				     rayContrib = F * g1;

				// Skip calculations if ray does not contribute much to the lighting
		
				if (luma(rayContrib) > 0.05){
				
					vec4 reflection = vec4(0.0,0.0,0.0,0.0);
					// Scale quality with ray contribution
					float rayQuality = 35*sqrt(luma(rayContrib));

					// Skip SSR if ray contribution is low
					if (rayQuality > 5.0) {
                    vec3 r = SSR(fragpos3.xyz, depth,normalize(normal2 + (roughness*3) * (normalize(p2) * poissonDisk[i].x + normalize(p3) * poissonDisk[i].y)), vec4(clamp((getSkyColorLut(L,sunPosition.xyz,L.y, temporals3Sampler).rgb),0,10),1));

								reflection.rgb = r;
         
                                reflection.a = 1.0;
         

                                    
					}

	
					indirectSpecular += (reflection.rgb * rayContrib);
					fresnelDiffuse += rayContrib;

				}
	
			}
          sunSpec = luma(specTerm);
        }
  
   
	// Day
	if (skyIntensity > 0.00001)
	{

		shadeDirS = clamp(skyIntensity*10,0,1)*dot(normal, sunPosition);
       	if(t1) shadeDirS = clamp(skyIntensity*10,0,1)*mix(max(phaseg(dot(view, sunPosition),sssAmount*0.4)*2, phaseg(dot(view, sunPosition),sssAmount*0.1))*3, shadeDirS, 0.35);
	}
	// Night
	if (skyIntensityNight > 0.00001)
	{
		shadeDirM = (skyIntensityNight*dot(normal, -sunPosition))*0.01;
       	if(t1) shadeDirM = clamp(skyIntensityNight,0,1)*mix(max(phaseg(dot(view, -sunPosition),0.45), phaseg(dot(view, -sunPosition),0.1)), shadeDirS, 0.35);
	

	}

    
        shadeDir =  clamp(shadeDirS + shadeDirM,0,1);
        
		shading = ambientLight + mix(vec3(0.0),direct, shadeDir);
        shading += (sunSpec*direct);
		ambientLight = mix(ambientLight*vec3(0.2,0.2,0.5)*2.0,ambientLight,1-rainStrength);	
		shading = mix(ambientLight,shading,1-rainStrength);	
 

        vec3 speculars  = (indirectSpecular/nSpecularSamples);
                                  speculars.rgb *= speculars.rgb;
                                  speculars.rgb *= 5.0;
        float mixweight = 1.0;
        if(postlight == 1) mixweight = 0.1;
		shading = mix(vec3(mixweight),shading,clamp((lmx)*5.0,0,1));
		shading = mix(shading,vec3(1.0),clamp((lmy),0,1));   

    vec3 dlight =   ( OutTexel * shading);
  dlight += (speculars*dlight); 
//    dlight = (indirectSpecular/nSpecularSamples + specTerm * direct.rgb) +  (1.0-fresnelDiffuse/nSpecularSamples) * dlight.rgb;
    if (light > 0.001)  dlight.rgb = OutTexel* pow(clamp((light*2)-0.2,0.0,1.0)/0.65*0.65+0.35,2.0);
    fragColor.rgb =  lumaBasedReinhardToneMapping(dlight)*clamp(ao,0.75,1.00);           		     
    if (light > 0.001)  fragColor.rgb *= clamp(vec3(2.0-shading*2)*light*2,1.0,10.0);


    float isWater = 0;
    if (texture(TranslucentSampler, texCoord).a *255 ==200) isWater = 1;
   
    if (isWater == 1){



    vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B)*fogcol.rgb;
    vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
    vec3 totEpsilon = dirtEpsilon*Dirt_Amount + waterEpsilon;
    fragColor.rgb *= clamp(exp(-length(fragpos)*totEpsilon),0.2,1.0);

    }


    	
float test = 0.0; 
   if(pbr.a*255 >1) test = 1.0;
 //		fragColor.rgb = clamp(vec3(pbr),0.01,1); 
    }




}
	else{

	 fragColor.rgb =  mix(reinhard_jodie(fragColor.rgb),fogcol.rgb*0.5,pow(depth,2048));
	}


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
