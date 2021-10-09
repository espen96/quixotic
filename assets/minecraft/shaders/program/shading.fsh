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

 in vec3 ambientUp;
 in vec3 ambientLeft;
 in vec3 ambientRight;
 in vec3 ambientB;
 in vec3 ambientF;
 in vec3 ambientDown;
 in vec3 avgSky;



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
in vec4 cloudx;
in float cloudy;
in float cloudz;
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

#define TORCH_R 1.0 
#define TORCH_G 0.5 
#define TORCH_B 0.2 

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FUDGE 32.0

#define Dirt_Amount 0.01 


#define Dirt_Mie_Phase 0.4  //Values close to 1 will create a strong peak around the sun and weak elsewhere, values close to 0 means uniform fog. 


#define Dirt_Absorb_R 0.65 
#define Dirt_Absorb_G 0.85 
#define Dirt_Absorb_B 1.05

#define Water_Absorb_R 0.25422
#define Water_Absorb_G 0.03751
#define Water_Absorb_B 0.01150

#define BASE_FOG_AMOUNT 0.2 //Base fog amount amount (does not change the "cloudy" fog)
#define CLOUDY_FOG_AMOUNT 1.0 
#define FOG_TOD_MULTIPLIER 1.0 //Influence of time of day on fog amount
#define FOG_RAIN_MULTIPLIER 1.0 //Influence of rain on fog amount

#define SSAO_SAMPLES 6

#define NORMDEPTHTOLERANCE 1.0



#define SSR_SAMPLES 5
#define SSR_MAXREFINESAMPLES 1
#define SSR_STEPINCREASE 0.1
#define SSR_IGNORETHRESH 0.9

#define NORMAL_SCATTER 0.006

#define CLOUDS_QUALITY 0.5

////////////////////////////////
    #define sssMin 22
    #define sssMax 47
    #define lightMin 48
    #define lightMax 72
    #define roughMin 73
    #define roughMax 157
    #define metalMin 158
    #define metalMax 251
//////////////////////////////////////////////////////////////////////////////////////////
vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}


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
//    if(rough < 0.001) rough = 0.1;

    pbr = vec4(emiss,sss,rough, metal);
 //   if(expanded > 170) pbr *=0;
    return pbr;    
}

/////////////////////////////////////////////////////////////////////////

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
	const int samples = 6;


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



vec3 lumaBasedReinhardToneMapping(vec3 color)
{
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma / (1. + luma);
	color *= toneMappedLuma / luma;
	color = pow(color, vec3(1. / 2.2));
	return color;
}



vec3 skyLut(vec3 sVector, vec3 sunVec,float cosT,sampler2D lut) {
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

uniform sampler2D FontSampler;  // ASCII 32x8 characters font texture unit

/*
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

float unpack8BitVec3IntoFloat(vec3 v, float min, float max) {
   float zeroTo24Bit = v.x + v.y * 256.0 + v.z * 256.0 * 256.0;
   float zeroToOne = zeroTo24Bit / 256.0 / 256.0 / 256.0;
   return zeroToOne * (max - min) + min;
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

// avoid hardware interpolation
vec4 sample_biquadratic_exact(sampler2D channel, vec2 uv) {
    vec2 res = (textureSize(channel,0).xy);
    vec2 q = fract(uv * res);
    ivec2 t = ivec2(uv * res);
    const ivec3 e = ivec3(-1, 0, 1);
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






float square(float x){
  return x*x;
}

float g(float NdotL, float roughness)
{
    float alpha = square(max(roughness, 0.02));
    return 2.0 * NdotL / (NdotL + sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotL)));
}



vec2 GGX_FV(float dotLH, float roughness)
{
	float alpha = roughness*roughness;

	// F
	float F_a, F_b;
	float dotLH5 = pow(1.0f-dotLH,5);
	F_a = 1.0f;
	F_b = dotLH5;

	// V
	float vis;
	float k = alpha/2.0f;
	float k2 = k*k;
	float invK2 = 1.0f-k2;
	vis = 1/(dotLH*dotLH*invK2 + k2);

	return vec2(F_a*vis,F_b*vis);
}

float GGX_D(float dotNH, float roughness)
{
	float alpha = roughness*roughness;
	float alphaSqr = alpha*alpha;
	float pi = 3.14159f;
	float denom = dotNH * dotNH *(alphaSqr-1.0) + 1.0f;

	float D = alphaSqr/(pi * denom * denom);
	return D;
}

float GGX(vec3 N, vec3 V, vec3 L, float roughness, float F0)
{
	vec3 H = normalize(V+L);

	float dotNL = clamp(dot(N,L),0,1);
	float dotLH = clamp(dot(L,H),0,1);
	float dotNH = clamp(dot(N,H),0,1);

	float D = GGX_D(dotNH,roughness);
	vec2 FV_helper = GGX_FV(dotLH,roughness);
	float FV = F0*FV_helper.x + (1.0f-F0)*FV_helper.y;
	float specular = dotNL * D * FV;

	return specular;
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

vec3 ComplexFresnel(float fresnel, float f0) {
    int metalidx = int(f0 * 255.0);
    vec3 k = vec3(1.0);
    vec3 n = vec3(0.0);
    float f = 1.0 - fresnel;

    vec3 k2 = k * k;
    vec3 n2 = n * n;
    float f2 = f * f;

    vec3 rs_num = n2 + k2 - 2 * n * f + f2;
    vec3 rs_den = n2 + k2 + 2 * n * f + f2;
    vec3 rs = rs_num / rs_den;
     
    vec3 rp_num = (n2 + k2) * f2 - 2 * n * f + 1;
    vec3 rp_den = (n2 + k2) * f2 + 2 * n * f + 1;
    vec3 rp = rp_num / rp_den;
    
    vec3 fresnel3 = clamp(0.5 * (rs + rp), vec3(0.0), vec3(1.0));
    fresnel3 *= fresnel3;

    return fresnel3;
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
//      vec3 rayDir     = reflect(normalize(fragpos.xyz), surfacenorm);
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
/*
        if (dtmp + SSR_IGNORETHRESH > fragdepth && dist < length(rayStep) * pow(length(rayRefine), 0.11) * 2.0) {
            refine++;
            if (refine >= SSR_MAXREFINESAMPLES)	break;
            rayRefine  -= rayStep;
            rayStep    *= SSR_STEPREFINE;
        }
*/
        rayStep        *= SSR_STEPINCREASE;
        rayRefine      += rayStep;
        rayPos          = rayStart+rayRefine;

    }


    vec4 candidate = vec4(0.0);
    if (fragdepth < dtmp + SSR_IGNORETHRESH && pos.y <= 1.0) {
        vec3 colortmp = texture(PreviousFrameSampler, pos.xy).rgb;



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

vec2 sphereToCarte(vec3 dir) {
    float lonlat = atan(-dir.x, -dir.z);
    return vec2(lonlat * (0.5/pi) +0.5,0.5*dir.y+0.5);
}
vec3 skyFromTex(vec3 pos,sampler2D sampler){
	vec2 p = sphereToCarte(pos);
	return texture(sampler,p*oneTexel*256.+vec2(18.5,1.5)*oneTexel).rgb;
}

float decodeFloat24(vec3 raw) {
    uvec3 scaled = uvec3(raw * 255.0);
    uint sign = scaled.r >> 7;
    uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
    uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
    return (-float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}
       vec3 FindNormal(sampler2D tex, vec2 uv, vec2 u)
            {
                    //u is one uint size, ie 1.0/texture size
                vec2 offsets[4];
					 offsets[0] = uv + vec2(-u.x, 0);
					 offsets[1] = uv + vec2(u.x, 0);
					 offsets[2] = uv + vec2(0, -u.x);
					 offsets[3] = uv + vec2(0, u.x);
               
                float hts[4];
                for(int i = 0; i < 4; i++)
                {
				

                    hts[i] = length(texture(tex, offsets[i]).x); 			

                }
               
                vec2 _step = vec2(0.1, 0.0);
               
			   
                vec3 va = normalize( vec3(_step.xy, hts[1]-hts[0]) );
                vec3 vb = normalize( vec3(_step.yx, hts[3]-hts[2]) );
				
	//            if (vtexcoord.x > 1.0 - 0.01 || vtexcoord.y > 1.0 - 0.01)  return vec3(0.0);   
	//            if (vtexcoord.x < 0.01 || vtexcoord.y < 0.01)              return vec3(0.0);
			   
                return cross(va,vb).rgb; //you may not need to swizzle the normal

               
            }
    vec3 viewToWorld(vec3 viewPos) {

    vec4 pos;
    pos.xyz = viewPos;
    pos.w = 0.0;
    pos = gbufferModelViewInverse * pos ;

    return pos.xyz;
}

void main() {


  	vec2 texCoord = texCoord; 
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
        float depth = deptha;   
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
    float depthtest = (deptha+depthb+depthc+depthd+depthe)/5;
    vec4 pbr = pbr( lmtrans,  unpackUnorm2x4((texture(DiffuseSampler, texCoord+vec2(oneTexel.y)).a)) );
    vec3 OutTexel = (texture(DiffuseSampler, texCoord).rgb);

vec3 test = vec3( (OutTexel));




    if( pbr.b *255 < 17) {
  float lum = luma(test);
  vec3 diff = test-lum;
 test = clamp(vec3(length(diff)),0.01,1);

 if(test.r >0.3) test *= 0.3;

 if(test.r < 0.05) test *= 5.0;
 if(test.r < 0.05) test *= 2.0;
 test = clamp(test*1.5-0.1,0,1);



    pbr.b = test.r;


    }
    float sssa = pbr.g;
    float ggxAmmount = pbr.b;
    float ggxAmmount2 = pbr.a;
    float light = pbr.r;
      
    
    OutTexel = toLinear(OutTexel);    

   fragColor.rgb = OutTexel;	
    if (depth > 1.0) light = 0;






    float lmx = 0;
    float lmy = 0;    
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    lmy = mix(lmtrans.y,(lmtrans2.y+lmtrans3.y+lmtrans4.y+lmtrans5.y)/4,res);
    lmx = mix((lmtrans2.y+lmtrans3.y+lmtrans4.y+lmtrans5.y)/4,lmtrans.y,res);

if(overworld == 1.0){
    






    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);


    vec3 suncol = texelFetch(temporals3Sampler,ivec2(8,37),0).rgb;

    vec3 np3 = normVec(view);

    vec3 direct = suncol;
    vec3 ambient;
		
    

 if (depthtest >= 1.0 || luma(OutTexel) == 0){


    vec3 atmosphere = ((skyLut(view,sunPosition.xyz,view.y,temporals3Sampler)))  ;

 		if (np3.y > 0.){
			atmosphere += stars(np3)*clamp(1-rainStrength,0,1);
//        	atmosphere += ((pow((1.0 / (1.0 + dot(-sunPosition, np3))),0.3)*suncol.rgb*0.05)*1)*clamp(1-(rainStrength),0,1);
            atmosphere += drawSun(dot(sunPosition,np3),0, suncol.rgb/150.,vec3(0.0))*clamp(1-rainStrength,0,1);
            atmosphere += drawSun(dot(-sunPosition,np3),0, suncol.rgb,vec3(0.0))*clamp(1-rainStrength,0,1);
            vec4 cloud = textureQuadratic(cloudsample, texCoord*CLOUDS_QUALITY);
            atmosphere = atmosphere*cloud.a+(cloud.rgb*1.1);
		}


    fragColor.rgb = reinhard(atmosphere) ;

    return;
 } 


    vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));


    float postlight = 1;

        if(lmx == 1) {
            lmx *= 0.75;
            postlight = 0.0;
            
        }

    vec3 lightmap = texture(temporals3Sampler,vec2(lmy,lmx)*(oneTexel*17)).xyz;
    if (light > 0.001)  lightmap.rgb = OutTexel* pow(clamp((light*10)-0.2,0.0,1.0)/0.65*0.65+0.35,10.0);

    if(postlight == 1)    OutTexel *= lightmap;

    if(lmx > 9.0)pbr = vec4(0.0);
    float ao = AmbientOcclusion(DiffuseDepthSampler,texCoord,Bayer256(gl_FragCoord.xy)) ;   

    // only do lighting if not sky and sunDir exists
    if (LinearizeDepth(depth) < far - FUDGE && length(sunPosition) > 0.99) {
	fragColor.a = 1;

        // first calculate approximate surface normal using depth map

        
        float depth2 = depthc;
        float depth3 = depthd;
        float depth4 = depthb;
        float depth5 = depthe;
        #define normalstrength  0.1;    
        float normaldistance = 2.0;    
        float normalpow = 4.0;    

        vec3 fragpos = backProject(vec4(scaledCoord, depth, 1.0)).xyz;
        fragpos.rgb += pow(luma(texture(DiffuseSampler,texCoord).rgb),normalpow)*normalstrength;

        vec3 p2 = backProject(vec4(scaledCoord + 2.0 * vec2(0.0, oneTexel.y), depth2, 1.0)).xyz;
        p2.rgb += pow(luma(texture(DiffuseSampler,texCoord + normaldistance*vec2(0.0, oneTexel.y)).rgb),normalpow)*normalstrength;
        p2 = p2 - fragpos;

        vec3 p3 = backProject(vec4(scaledCoord + 2.0 * vec2(oneTexel.x, 0.0), depth3, 1.0)).xyz;
            p3.rgb += pow(luma(texture(DiffuseSampler,texCoord + normaldistance* vec2(oneTexel.x, 0.0)).rgb),normalpow)*normalstrength;

        p3 = p3 - fragpos;
        vec3 p4 = backProject(vec4(scaledCoord - 2.0 * vec2(0.0, oneTexel.y), depth4, 1.0)).xyz;
                    p4.rgb += pow(luma(texture(DiffuseSampler,texCoord - normaldistance* vec2(0.0, oneTexel.y)).rgb),normalpow)*normalstrength;

        p4 = p4 - fragpos;
        vec3 p5 = backProject(vec4(scaledCoord - 2.0 * vec2(oneTexel.x, 0.0), depth5, 1.0)).xyz;
                    p5.rgb += pow(luma(texture(DiffuseSampler,texCoord - normaldistance* vec2(oneTexel.x, 0.0)).rgb),normalpow)*normalstrength;
                    
        p5 = p5 - fragpos;
        vec3 normal = normalize(cross( p2,  p3)) 
                    + normalize(cross(-p4,  p3)) 
                    + normalize(cross( p2, -p5)) 
                    + normalize(cross(-p4, -p5));
        normal = normal == vec3(0.0) ? vec3(0.0, 1.0, 0.0) : normalize(-normal);


	    vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.));

		vec3 ambientLight  = ambientUp   *mix(clamp( ambientCoefs.y,0.,1.), 0.166, sssa);
             ambientLight += ambientDown*1.5 *mix(clamp(-ambientCoefs.y,0.,1.), 0.166, sssa);
             ambientLight += ambientRight*mix(clamp( ambientCoefs.x,0.,1.), 0.166, sssa);
             ambientLight += ambientLeft *mix(clamp(-ambientCoefs.x,0.,1.), 0.166, sssa);
             ambientLight += ambientB    *mix(clamp( ambientCoefs.z,0.,1.), 0.166, sssa);
             ambientLight += ambientF    *mix(clamp(-ambientCoefs.z,0.,1.), 0.166, sssa);
             ambientLight *= (1.0+rainStrength*0.2);
//             ambientLight *= 1.5;
    
	


    bool isSSS = sssa > 0.0;
    vec2 poissonDisk[32];
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



	vec3 shading;
	float shadeDir = 0;
	float shadeDirS = 0;
	float shadeDirM = 0;
    vec3 sunPosition2 = mix(sunPosition,-sunPosition,clamp(skyIntensityNight*3,0,1));


		shadeDirS  = clamp(skyIntensity,0,1)*dot(normal, sunPosition2);
       	shadeDirS +=(clamp(skyIntensity,0,1)   *  mix(max(phaseg(dot(view, sunPosition2),sssa*0.4)*2, phaseg(dot(view, sunPosition2),sssa*0.1))*3, shadeDirS, 0.35))*float(isSSS);
           

		shadeDirM  = clamp(skyIntensityNight,0,1)*dot(normal, sunPosition2);
       	shadeDirM +=(clamp(skyIntensityNight,0,1)*mix(max(phaseg(dot(view, sunPosition2),sssa*0.4)*2, phaseg(dot(view, sunPosition2),sssa*0.1))*3, shadeDirS, 0.35))*float(isSSS);
	 

	

			vec3 f0 = vec3(0.04);
    

			float roughness = 1-ggxAmmount;
	
			vec3 indirectSpecular = vec3(0.0);
         
			const int nSpecularSamples = 16;



        if(ggxAmmount2 > 0.001){ 
            f0 = vec3(0.8);  
            ggxAmmount = ggxAmmount2;
            vec3 normal2 = normalize(worldToView(normal) );
         
			mat3 basis = CoordBase(normal2);
			vec3 normSpaceView = -np3*basis;
			vec3 rayContrib = vec3(0.0);
			vec3 reflection = vec3(0.0);    
            float wdepth = texture(TranslucentDepthSampler, texCoord).r;

            float ldepth = LinearizeDepth(wdepth);
            vec3 fragpos3 = (gbPI * vec4(texCoord, ldepth, 1.0)).xyz;
            fragpos3 *= ldepth;
  

    float wdepth2 = texture(TranslucentDepthSampler, texCoord + vec2(0.0, oneTexel.y)).r;
    float wdepth3 = texture(TranslucentDepthSampler, texCoord + vec2(oneTexel.x, 0.0)).r;
    float ldepth2 = LinearizeDepth(wdepth2);
    float ldepth3 = LinearizeDepth(wdepth3);
    ldepth2 = abs(ldepth - ldepth2) > NORMDEPTHTOLERANCE ? ldepth : ldepth2;
    ldepth3 = abs(ldepth - ldepth3) > NORMDEPTHTOLERANCE ? ldepth : ldepth3;



        vec3 fragpos = (gbPI * vec4(texCoord, ldepth, 1.0)).xyz;
        fragpos *= ldepth;
        vec3 p8 = (gbPI * vec4(texCoord + vec2(0.0, oneTexel.y), ldepth2, 1.0)).xyz;
        p8 *= ldepth2;
        vec3 p7 = (gbPI * vec4(texCoord + vec2(oneTexel.x, 0.0), ldepth3, 1.0)).xyz;
        p7 *= ldepth3;
        vec3 normal3 = -normalize(cross(p8 - fragpos, p7 - fragpos));
        
        float ndlsq = dot(normal3, vec3(0.0, 0.0, 1.0));
                float horizon = clamp(ndlsq * 100000.0, -1.0, 1.0);

	    vec4 reflection2 = vec4(0.0);
			for (int i = 0; i < nSpecularSamples; i++){

				
				


                    vec3 r = SSR(fragpos3.xyz, depth,normalize(normal2 + (1-ggxAmmount) * (normalize(p2) * poissonDisk[i].x + normalize(p3) * poissonDisk[i].y)), vec4(avgSky,1));
								reflection2.rgb += r;
                                reflection2.a += 1.0;
					

	

				
	
			}

                        reflection2 = reflection2 / nSpecularSamples;    

		float fresnel = pow(clamp(1.0 + dot(normal3, normalize(fragpos3.xyz)), 0.0, 1.0), 5.0);


        float lookfresnel = clamp(exp(-13 * clamp(ndlsq * horizon, 0.0, 1.0) + 3.0)*100, 0.0, 1.0);
        vec4 color2 = vec4(OutTexel,1);



//           reflection2 = clamp(reflection2,0,10);           
           reflection2 = mix(vec4(avgSky*0.7,1),reflection2,luma(reflection2.rgb));
					indirectSpecular += ((reflection2.rgb )*(fresnel*OutTexel));
        OutTexel *= 0.25;

        }
        float sunSpec = ((GGX(normal,-normalize(view),  sunPosition2, 1-ggxAmmount, f0.x)));		
  

   
        float mixweight = 0.1;

    
        shadeDir =  clamp(shadeDirS + shadeDirM,0,1);
        
		shading = ambientLight + mix(vec3(0.0),direct, shadeDir);
        shading += (sunSpec*direct);
        
		ambientLight = mix(ambientLight*vec3(0.2,0.2,0.5)*2.0,ambientLight,1-rainStrength);	
        if(postlight == 1){ambientLight = mix(vec3(0.1,0.1,0.5),vec3(1.0),1-rainStrength);
        mixweight = 1.0;
        }
		shading = mix(ambientLight,shading,1-rainStrength);	
 

        vec3 speculars  = (indirectSpecular);

		shading = mix(vec3(mixweight),shading,clamp((lmx)*5.0,0,1));
		shading = mix(shading,vec3(1.0),clamp((lmy),0,1));   

    vec3 dlight =   ( OutTexel * clamp(shading,0.1,10));
    dlight += (speculars); 
    fragColor.rgb =  lumaBasedReinhardToneMapping(dlight*clamp(ao,0.75,1.00));           		     
    if (light > 0.001)  fragColor.rgb *= clamp(vec3(2.0-shading*2)*light*2,1.0,10.0);


    float isWater = 0;
    if (texture(TranslucentSampler, texCoord).a *255 ==200) isWater = 1;
   
    if (isWater == 1){



    vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B)*fogcol.rgb;
    vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
    vec3 totEpsilon = dirtEpsilon*Dirt_Amount + waterEpsilon;
    fragColor.rgb *= clamp(exp(-length(fragpos)*totEpsilon),0.2,1.0);

    }


//		fragColor.rgb = clamp(vec3(pbr),0.01,1);     


}





}
	else{

	 fragColor.rgb =  mix(reinhard_jodie(fragColor.rgb*( (((lmx+ 0.15)*fogcol.rgb)+((lmy*lmy*lmy)*vec3(TORCH_R,TORCH_G,TORCH_G))))),fogcol.rgb*0.5,pow(depth,2048));
         if (light > 0.001)  fragColor.rgb *= clamp(vec3(2.0-1*2)*light*2,1.0,10.0);
	}


/*
	vec4 numToPrint = vec4(decodeFloat24(cloudx.xyz));

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
