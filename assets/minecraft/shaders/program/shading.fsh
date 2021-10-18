#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D temporals3Sampler;
uniform sampler2D cloudsample;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D PreviousFrameSampler;

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
 in vec3 suncol;
 in vec3 upPosition;
 in float worldTime;


in vec2 oneTexel;
in vec3 sunDir;
in vec4 fogcol;

in vec4 rain;

in mat4 wgbufferModelView;

in vec2 texCoord;

 in mat4 gbufferModelViewInverse;
 in mat4 gbufferModelView;

 in mat4 gbufferProjection;

in float near;
in float far;
in float end;
in float overworld;
in float aspectRatio;

in float sunElevation;
in float rainStrength;
in vec3 sunVec;


in vec3 sunPosition;
in vec3 sunPosition3;
in float skyIntensity;
in float skyIntensityNight;



out vec4 fragColor;
mat4 gbufferProjectionInverse = inverse(gbufferProjection);
mat4 wgbufferModelViewInverse = inverse(wgbufferModelView);


#define AOStrength 1.0
#define radius 1.0
#define steps 6

#define TORCH_R 1.0 
#define TORCH_G 0.7 
#define TORCH_B 0.5 

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
float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}

vec4 pbr (vec2 in1,vec2 in2, vec3 test ){

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


    return pbr;    
}

/////////////////////////////////////////////////////////////////////////

vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}

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

float decodeFloat7_4(uint raw) {
    uint sign = raw >> 11u;
    uint exponent = (raw >> 7u) & 15u;
    uint mantissa = 128u | (raw & 127u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 14.0);
}

float decodeFloat6_4(uint raw) {
    uint sign = raw >> 10u;
    uint exponent = (raw >> 6u) & 15u;
    uint mantissa = 64u | (raw & 63u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 13.0);
}

vec3 decodeColor(vec4 raw) {
    uvec4 scaled = uvec4(round(raw * 255.0));
    uint encoded = (scaled.r << 24) | (scaled.g << 16) | (scaled.b << 8) | scaled.a;
    
    return vec3(
        decodeFloat7_4(encoded >> 21),
        decodeFloat7_4((encoded >> 10) & 2047u),
        decodeFloat6_4(encoded & 1023u)
    );
}

uint encodeFloat7_4(float val) {
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 14.0)) & 127u;
    return (sign << 11u) | (exponent << 7u) | mantissa;
}

uint encodeFloat6_4(float val) {
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 13.0)) & 63u;
    return (sign << 10u) | (exponent << 6u) | mantissa;
}

vec4 encodeColor(vec3 color) {
    uint r = encodeFloat7_4(color.r);
    uint g = encodeFloat7_4(color.g);
    uint b = encodeFloat6_4(color.b);
    
    uint encoded = (r << 21) | (g << 10) | b;
    return vec4(
        encoded >> 24,
        (encoded >> 16) & 255u,
        (encoded >> 8) & 255u,
        encoded & 255u
    ) / 255.0;
}


float invLinZ (float lindepth){
	return -((2.0*near/lindepth)-far-near)/(far-near);
}
float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}
float linZ2(float depth) 
{
    return (2.0 * near * far) / (far + near - depth * (far - near));    
}



vec4 backProject(vec4 vec) {
    vec4 tmp = wgbufferModelViewInverse * vec;
    return tmp / tmp.w;
}


vec3 normVec (vec3 vec){
	return vec*inversesqrt(dot(vec,vec));
}

float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0/1.6180339887 * Time);
}


vec3 lumaBasedReinhardToneMapping(vec3 color)
{
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma / (1. + luma);
	color *= clamp(toneMappedLuma / luma,0,10);
	color = pow(color, vec3(1. / 2.2));
	return color;
}


#define DOWNSCALE 32.0

vec4 interpolate_bilinear(vec2 p, vec4 q[16])
{
    vec4 r1 = (1.0-p.x)*q[5]+p.x*q[9];
    vec4 r2 = (1.0-p.x)*q[6]+p.x*q[10];
    return (1.0-p.y)*r1+p.y*r2;
}

vec3 skyLut(vec3 sVector, vec3 sunVec,float cosT,sampler2D lut) {
	const vec3 moonlight = vec3(0.8, 1.1, 1.4) * 0.06;

	float mCosT = clamp(cosT,0.0,1.);
	float cosY = dot(sunVec,sVector);
	float x = ((cosY*cosY)*(cosY*0.5*256.)+0.5*256.+18.+0.5)*oneTexel.x;
	float y = (mCosT*256.+1.0+0.5)*oneTexel.y;

	vec2 uv = vec2(x,y);
  	vec2 lowres = ScreenSize.xy/0.75;
  	vec2 pixel = 1.0/lowres;
  	vec2 base_uv = floor(uv*lowres)/lowres;
  	vec2 sub_uv = fract(uv*lowres);

    vec4 q[16];
    
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            q[i*4+j] = texture(lut,base_uv+vec2(pixel.x*float(i-1),pixel.y*float(j-1)));
        }
    }
        vec4 bicubic = interpolate_bilinear(sub_uv,q);

	return (bicubic.xyz);


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


vec3 worldToView(vec3 worldPos) {

    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos +gbufferModelView[3];

    return pos.xyz;
}

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + vec4(0,0,-1.0,10);
    return fragposition.xyz / fragposition.w;
}

vec3 nvec3(vec4 pos) {
    return pos.xyz/pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

float cdist(vec2 coord) {
	return max(abs(coord.x - 0.5), abs(coord.y - 0.5)) * 1.85;
}

vec4 Raytrace(sampler2D depthtex, vec3 viewPos, vec3 normal, float dither, out float border, 
			  float maxf, float stp, float ref, float inc) {
	vec3 pos = vec3(0.0);
	float dist = 0.0;
    int sr = 0;


	vec3 start = viewPos;

    vec3 vector = stp * reflect(normalize(viewPos), normalize(normal));
    viewPos += vector;
	vec3 tvector = vector;



    for(int i = 0; i < 12; i++) {
        pos = nvec3(gbufferProjection * nvec4(viewPos)) * 0.5 + 0.5;
		if (pos.x < -0.05 || pos.x > 1.05 || pos.y < -0.05 || pos.y > 1.05) break;

		vec3 rfragpos = vec3(pos.xy, texture(DiffuseDepthSampler,pos.xy).r);
        rfragpos = nvec3(gbufferProjectionInverse * nvec4(rfragpos * 2.0 - 1.0));
		dist = length(start - rfragpos);

        float err = length(viewPos - rfragpos);
		if (err < pow(length(vector) * pow(length(tvector), 0.11), 1.1) * 1.2) {
			sr++;
			if (sr >= maxf) break;
			tvector -= vector;
			vector *= ref;
		}
        vector *= inc;
        tvector += vector;
		viewPos = start + tvector * (dither * 0.05 + 0.975);
    }

	border = cdist(pos.st);


	return vec4(pos, dist);
}
vec4 SSR(vec3 fragpos, float fragdepth, vec3 surfacenorm, vec4 skycol, float noise) {

    vec3 pos    = vec3(0.0);




    vec4 color = vec4(0.0);
	float border = 0.0;
     pos = Raytrace(DiffuseDepthSampler, fragpos, surfacenorm,  noise, border, 4, 1.0, 0.1, 2.0).xyz;

	border = clamp(13.333 * (1.0 - border), 0.0, 1.0);
	
	if (pos.z < 1.0 - 1e-5) {
		color.a = texture(PreviousFrameSampler, pos.st).a;
		if (color.a > 0.001) color.rgb = texture(PreviousFrameSampler, pos.st).rgb;

		
		color.a *= border;
	}

    return color;
}

vec3 reinhard_jodie(vec3 v)
{
    float l = luma(v);
    vec3 tv = v / (1.0f + v);
    tv = mix(v / (1.0f + l), tv, tv);
    return 	pow(tv, vec3(1. / 2.2));
}


float decodeFloat24(vec3 raw) {
    uvec3 scaled = uvec3(raw * 255.0);
    uint sign = scaled.r >> 7;
    uint exponent = ((scaled.r >> 1u) & 63u) - 31u;
    uint mantissa = ((scaled.r & 1u) << 16u) | (scaled.g << 8u) | scaled.b;
    return (-float(sign) * 2.0 + 1.0) * (float(mantissa) / 131072.0 + 1.0) * exp2(float(exponent));
}	
vec3 toScreenSpace(vec2 p) {
		vec4 fragposition = gbufferProjectionInverse * vec4(vec3(p, texture2D(DiffuseDepthSampler, p).x) * 2.0 - 1.0, 1.0);
		return fragposition.xyz /= fragposition.w;
	}
	int bitfieldReverse(int a) {
		a = ((a & 0x55555555) << 1 ) | ((a & 0xAAAAAAAA) >> 1);
		a = ((a & 0x33333333) << 2 ) | ((a & 0xCCCCCCCC) >> 2);
		a = ((a & 0x0F0F0F0F) << 4 ) | ((a & 0xF0F0F0F0) >> 4);
		a = ((a & 0x00FF00FF) << 8 ) | ((a & 0xFF00FF00) >> 8);
		a = ((a & 0x0000FFFF) << 16) | ((a & 0xFFFF0000) >> 16);
		return a;
	}

	#define hammersley(i, N) vec2( float(i) / float(N), float( bitfieldReverse(i) ) * 2.3283064365386963e-10 )
	#define tau 6.2831853071795864769252867665590
	#define circlemap(p) (vec2(cos((p).y*tau), sin((p).y*tau)) * p.x)
	float jaao(vec2 p, vec3 normal, float noise) {

		// By Jodie. With some modifications

		float ao = 1.0;

		vec3 p3 = toScreenSpace(p);
		vec2 clipRadius = radius * vec2(ScreenSize.x / ScreenSize.y, 1.0) / length(p3);

		vec3 v = normalize(-p3);

		float nvisibility = 0.0;
		float vvisibility = 0.0;

		for (int i = 0; i < steps; i++) {

			vec2 circlePoint = circlemap(hammersley(i * 15 + 1, 16 * steps)) * clipRadius;

			circlePoint *= noise + 0.1;

			vec3 o  = toScreenSpace(circlePoint    +p) - p3;
			vec3 o2 = toScreenSpace(circlePoint*.25+p) - p3;
			float l  = length(o );
			float l2 = length(o2);
			o /=l ;
			o2/=l2;

			nvisibility += clamp(1.-max(
				dot(o , normal) - clamp((l -radius)/radius,0.,1.),
				dot(o2, normal) - clamp((l2-radius)/radius,0.,1.)
			), 0., 1.);

			vvisibility += clamp(1.-max(
				dot(o , v) - clamp((l -radius)/radius,0.,1.),
				dot(o2, v) - clamp((l2-radius)/radius,0.,1.)
			), 0., 1.);

		}

		ao = min(vvisibility * 2.0, nvisibility) / float(steps);


		return ao;

	}


#define  projMAD2(m, v) (diagonal3(m) * (v) + vec3(0,0,m[3].b))

vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD2(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}


float rayTraceShadow(vec3 dir,vec3 position,float dither){

    const float quality = 15.0;
    vec3 clipPosition = nvec3(gbufferProjection * nvec4(position)) * 0.5 + 0.5;
	//prevents the ray from going behind the camera
	float rayLength = ((position.z + dir.z * far*sqrt(3.)) > -near) ? (-near -position.z) / dir.z : far*sqrt(3.);
    //vec3 direction = ((position+dir*rayLength))-clipPosition;  //convert to clip space
    vec3 direction = toClipSpace3(position+dir*rayLength)-clipPosition;
    direction.xyz = direction.xyz/max(abs(direction.x)/oneTexel.x,abs(direction.y)/oneTexel.y);	//fixed step size




    vec3 stepv = direction *15.0;

	vec3 spos = clipPosition+stepv;


	for (int i = 0; i < int(quality); i++) {
		spos += stepv*dither;

		float sp = texture2D(DiffuseDepthSampler,spos.xy).x;
        if( sp < spos.z) {

			float dist = abs(linZ(sp)-linZ(spos.z))/linZ(spos.z);

			if (dist < 0.05 ) return exp2(position.z/4.);

	}

	}
    return 1.0;
}
// simplified version of joeedh's https://www.shadertoy.com/view/Md3GWf
// see also https://www.shadertoy.com/view/MdtGD7

// --- checkerboard noise : to decorelate the pattern between size x size tiles 

// simple x-y decorrelated noise seems enough
#define stepnoise0(p, size) rnd( floor(p/size)*size ) 
#define rnd(U) fract(sin( 1e3*(U)*mat2(1,-7.131, 12.9898, 1.233) )* 43758.5453)

//   joeedh's original noise (cleaned-up)
vec2 stepnoise(vec2 p, float size) { 
    p = floor((p+10.)/size)*size;          // is p+10. useful ?   
    p = fract(p*.1) + 1. + p*vec2(2,3)/1e4;    
    p = fract( 1e5 / (.1*p.x*(p.y+vec2(0,1)) + 1.) );
    p = fract( 1e5 / (p*vec2(.1234,2.35) + 1.) );      
    return p;    
}

// --- stippling mask  : regular stippling + per-tile random offset + tone-mapping

#define SEED1 1.705
#define DMUL  8.12235325       // are exact DMUL and -.5 important ?

float mask(vec2 p) { 

    p += ( stepnoise0(p, 5.5) - .5 ) *DMUL;   // bias [-2,2] per tile otherwise too regular
    float f = fract( p.x*SEED1 + p.y/(SEED1+.15555) ); //  weights: 1.705 , 0.5375

    //return f;  // If you want to skeep the tone mapping
    f *= 1.03; //  to avoid zero-stipple in plain white ?

    // --- indeed, is a tone mapping ( equivalent to do the reciprocal on the image, see tests )
    // returned value in [0,37.2] , but < 0.57 with P=50% 

    return  (pow(f, 150.) + 1.3*f ) / 2.3; // <.98 : ~ f/2, P=50%  >.98 : ~f^150, P=50%    
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
float OrenNayar(vec3 normal, vec3 viewvec, vec3 lvec, float roughness) {    //qualitative model

    vec3 h  = normalize(viewvec + lvec);

    float nDotL     = clamp(dot(normal, lvec),0,1);

    float nDotV     = clamp(dot(normal, viewvec),0,1);

    float t     = max(nDotL, nDotV);
    float g     = max(dot(viewvec-normal*nDotV, lvec-normal*nDotL), 0.0);
    float c     = g*(1/t) - g*t;
    float a     = 0.285 * (1/(roughness + 0.57)) + 0.5;
    float b     = 0.45 * roughness * (1/(roughness + 0.09));

    float on    = max(nDotL, 0.0) * (b * c + a);

    return max(on, 0.0);
}
void main() {
    //float noise = mask(gl_FragCoord.xy+(Time*100));
    float noise = R2_dither();
    vec4 outcol = vec4(0.0);
    vec2 lmtrans = unpackUnorm2x4((texture(DiffuseSampler, texCoord).a));
    float depth = texture(TranslucentDepthSampler, texCoord).r;
    //if(depth >= 1) lmtrans = vec2(0.0); 
    vec2 lmtrans2 = unpackUnorm2x4((texture(DiffuseSampler, texCoord-vec2(0,oneTexel.y)).a));
    float depthb = texture(TranslucentDepthSampler, texCoord-vec2(0,oneTexel.y)).r;
    lmtrans2 *= 1-(depthb -depth);

    vec2 lmtrans3 = unpackUnorm2x4((texture(DiffuseSampler, texCoord+vec2(0,oneTexel.y)).a));
    float depthc = texture(TranslucentDepthSampler, texCoord+vec2(0,oneTexel.y)).r;
    lmtrans3 *= 1-(depthc -depth);

    vec2 lmtrans4 = unpackUnorm2x4((texture(DiffuseSampler, texCoord+vec2(oneTexel.x,0)).a));
    float depthd = texture(TranslucentDepthSampler, texCoord+vec2(oneTexel.x,0)).r;
    lmtrans4 *= 1-(depthd -depth);

    vec2 lmtrans5 = unpackUnorm2x4((texture(DiffuseSampler, texCoord-vec2(oneTexel.x,0)).a));
    float depthe = texture(DiffuseDepthSampler, texCoord-vec2(oneTexel.x,0)).r;
    lmtrans5 *= 1-(depthe -depth);

    bool isEyeInWater = (fogcol.a > 0.078 && fogcol.a < 0.079 );
    bool isEyeInLava = (fogcol.r ==0.6 && fogcol.b == 0.0 );
 


	if(overworld != 1.0 && end != 1.0){


    vec2 p_m = texCoord;
    vec2 p_d = p_m;
    p_d.xy -= Time * 0.1;
    vec2 dst_map_val = vec2(Nnoise(p_d.xy));
    vec2 dst_offset = dst_map_val.xy;

    dst_offset *= 2.0;

    dst_offset *= 0.0025;
	
    //reduce effect towards Y top
	
    dst_offset *= (1. - p_m.t);	

	vec2 texCoord = p_m.st + dst_offset;



	}



    vec3 OutTexel = (texture(DiffuseSampler, texCoord).rgb);

    vec4 pbr = pbr( lmtrans,  unpackUnorm2x4((texture(DiffuseSampler, texCoord+vec2(oneTexel.y)).a)),OutTexel );






    float sssa = pbr.g;
    float ggxAmmount = pbr.b;
    float ggxAmmount2 = pbr.a;
    float light = pbr.r;
      
    
    OutTexel = toLinear(OutTexel);    


    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    float lmy = mix(lmtrans.y,(lmtrans2.y+lmtrans3.y+lmtrans4.y+lmtrans5.y)/4,res);
    float lmx = mix((lmtrans2.y+lmtrans3.y+lmtrans4.y+lmtrans5.y)/4,lmtrans.y,res);

if(overworld == 1.0){


    vec3 screenPos = vec3(texCoord, depth);
    vec3 clipPos = screenPos * 2.0 - 1.0;
    vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
    vec3 viewPos = tmp.xyz / tmp.w;	
	vec3 p3 = mat3(gbufferModelViewInverse) * viewPos;



    vec3 view = normVec(p3);

    vec3 direct = suncol;
    vec3 ambient;
    //float depthtest = (depth+depthb+depthc+depthd+depthe)/5;
		

	bool sky = depth >= 1.0;

     if (sky){


        vec3 atmosphere = ((skyLut(view,sunPosition3.xyz,view.y,temporals3Sampler)))  ;

 		if (view.y > 0.){
			atmosphere += stars(view)*clamp(1-rainStrength,0,1);
            atmosphere += drawSun(dot(sunPosition3,view),0, suncol.rgb/150.,vec3(0.0))*clamp(1-rainStrength,0,1)*20;
            atmosphere += drawSun(dot(-sunPosition3,view),0, suncol.rgb,vec3(0.0))*clamp(1-rainStrength,0,1);
            vec4 cloud = texture(cloudsample, texCoord*CLOUDS_QUALITY);
            atmosphere = atmosphere*cloud.a+(cloud.rgb);
		}

  	atmosphere= (clamp(atmosphere*1.1,0,2));
    outcol.rgb = reinhard(atmosphere) ;



    }
    else{ 


    vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));


    float postlight = 1;

        if(lmx > 0.95) {
            lmx *= 0.75;
            lmy = 0.1;
            postlight = 0.0;
            
        }

    vec3 lightmap = texture(temporals3Sampler,vec2(lmy,lmx)*(oneTexel*17)).xyz;

    if(postlight == 1)    OutTexel *= lightmap;






        
        float depth2 = depthc;
        float depth3 = depthd;
        float depth4 = depthb;
        float depth5 = depthe;
        const float normalstrength = 0.1;    
        const float normaldistance = 2.5;    
        const float normalpow = 4.0;    

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
      vec3 normal3 = worldToView (normal);


    bool isWater = (texture(TranslucentSampler, texCoord).a *255 ==200);


	    vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.0));

            vec3 ambientLight = ambientUp*clamp(ambientCoefs.y,0.,1.);
            ambientLight += ambientDown*clamp(-ambientCoefs.y,0.,1.);
            ambientLight += ambientRight*clamp(ambientCoefs.x,0.,1.);
            ambientLight += ambientLeft*clamp(-ambientCoefs.x,0.,1.);
            ambientLight += ambientB*clamp(ambientCoefs.z,0.,1.);
            ambientLight += ambientF*clamp(-ambientCoefs.z,0.,1.);
            //ambientLight = avgSky;
            ambientLight *= (1.0+rainStrength*0.2);
            ambientLight *= 1.75;

            ambientLight = clamp(ambientLight * (pow(lmx,8.0)*1.5) + lmy*vec3(TORCH_R,TORCH_G,TORCH_B),0,2.0) ;

	




	vec3 shading= vec3(0.0);

    vec3 sunPosition2 = mix(sunPosition3,-sunPosition3,clamp(skyIntensityNight*3,0,1));
    vec3 sunVec = mix(sunVec,-sunVec,clamp(skyIntensityNight*3,0,1));
 
	//float shadeDir  = clamp(dot(normal, sunPosition2),0,1);
    float shadeDir = OrenNayar( normal, p3, sunPosition2, 1-ggxAmmount);
    shadeDir+= (mix(max(phaseg(dot(view, sunPosition2),0.45)*1.5, phaseg(dot(view, sunPosition2),0.1))*3, shadeDir, 0.35))*float(sssa)*lmx;
    shadeDir = clamp(shadeDir,0,1);
    vec3 f0 = vec3(0.04);
	float roughness = 1-ggxAmmount;
	vec3 speculars = vec3(0.0);



            
    if(ggxAmmount2 > 0.001){ 
            f0 = vec3(0.8);  
            ggxAmmount = ggxAmmount2;
            float ldepth = linZ2(depth);

			vec3 reflection = vec3(0.0);    
            vec3 fragpos3 = (vec4(texCoord, ldepth, 1.0)).xyz;
            fragpos3 *= ldepth;

            vec4 reflection2 = vec4(SSR(viewPos.xyz, depth,normal3, vec4(avgSky,1),noise));	

            float fresnel = pow(clamp(1.0 + dot(normal3, normalize(fragpos3.xyz)), 0.0, 1.0), 5.0);


            vec4 color2 = vec4(OutTexel,1);
            reflection2 = mix(vec4(avgSky,1),reflection2,reflection2.a);
            speculars += ((reflection2.rgb )*(fresnel*OutTexel));
            OutTexel *= 0.1;

    }
        
        float sunSpec = ((GGX(normal,-normalize(view),  sunPosition2, 1-ggxAmmount, f0.x)));		

        
        float screenShadow = (rayTraceShadow(sunVec,viewPos,noise)*clamp((lmx-0.0)*1,0,1));
        screenShadow = mix(screenShadow,((screenShadow+lmy),0,1),clamp((lmy),0,1));
        shadeDir *= screenShadow;
        
        shadeDir = mix(0.0,shadeDir,clamp((lmx)*5.0,0,1));
      
		shading = ambientLight + (direct*shadeDir);
        shading += (sunSpec*direct)*shadeDir;      

        shading += lightmap*0.1;
  
        
		ambientLight = mix(ambientLight*vec3(0.2,0.2,0.5)*2.0,ambientLight,1-rainStrength);	
        if(postlight == 1)ambientLight = mix(vec3(0.1,0.1,0.5),vec3(1.0),1-rainStrength);

        
		shading = mix(ambientLight,shading,1-rainStrength);	
        if (light > 0.001)  shading.rgb = vec3(light*2.0);


    vec3 dlight =   ( OutTexel * clamp(shading,0.1,10))+speculars;



	float ao = 1.0 *((1.0 - AOStrength) + jaao(texCoord,normal3,noise) * AOStrength);
    
    outcol.rgb =  lumaBasedReinhardToneMapping(dlight*ao);           	
    	     
    outcol.rgb *= 1.0+max(0.0,light);




    if (isWater){



    vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B)*fogcol.rgb;
    vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
    vec3 totEpsilon = dirtEpsilon*Dirt_Amount + waterEpsilon;
    outcol.rgb *= clamp(exp(-length(viewPos)*totEpsilon),0.2,1.0);

    }	

    outcol.a = 1.0;
    // 	outcol.rgb = clamp(vec3(shadeDir),0.01,1);     


}





}
	else{

	 outcol.rgb =  mix(reinhard_jodie(OutTexel.rgb*( (((lmx+ 0.15)*fogcol.rgb)+((lmy*lmy*lmy)*vec3(TORCH_R,TORCH_G,TORCH_G))))),fogcol.rgb*0.5,pow(depth,2048));
         if (light > 0.001)  outcol.rgb *= clamp(vec3(2.0-1*2)*light*2,1.0,10.0);
	}
    outcol= (clamp(outcol,0,2));

    fragColor = (outcol.rgba);


/*
	vec4 numToPrint = vec4(worldTime);
	// Define text to draw
    clearTextBuffer();
    c('R'); c(':'); c(' '); floatToDigits(numToPrint.r);
    printTextAt(1.0, 1.0);

    clearTextBuffer();
    c('G'); c(':'); c(' '); floatToDigits(numToPrint.g);
    printTextAt(1.0, 2.0);

    clearTextBuffer();
    c('B'); c(':'); c(' '); floatToDigits(numToPrint.b);
    printTextAt(1.0, 3.0);

    clearTextBuffer();
    c('A'); c(':'); c(' '); floatToDigits(numToPrint.a);
    printTextAt(1.0, 4.0);

    fragColor += colour;
*/



}
