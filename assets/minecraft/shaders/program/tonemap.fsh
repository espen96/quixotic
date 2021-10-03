#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D MainSampler;
uniform sampler2D BloomSampler;
uniform sampler2D blursampler;
uniform vec2 ScreenSize;
out vec4 fragColor;

in vec2 texCoord;




    #define EXPOSURE 1.42 
    #define TONEMAP_WHITE_CURVE 1.7 
    #define TONEMAP_LOWER_CURVE 1.2 
    #define TONEMAP_UPPER_CURVE 1.3 
    #define CROSSTALK 0.25 // Desaturates bright colors and preserves saturation in darker areas (inverted if negative). Helps avoiding almsost fluorescent colors 
    #define SATURATION 0.25 // Negative values desaturates colors, Positive values saturates color, 0 is no change
    #define ndeSat 7.0
    #define Purkinje_strength 1.0	// Simulates how the eye is unable to see colors at low light intensities. 0 = No purkinje effect at low exposures 
    #define Purkinje_R 0.4
    #define Purkinje_G 0.7 
    #define Purkinje_B 1.0
    #define Purkinje_Multiplier 0.1 // How much the purkinje effect increases brightness
    #define SAMPLE_OFFSET 5.
    #define INTENSITY 0.1









precision highp float;

#define PI 3.1415926535897932384626433832795
#define EULER 2.7182818284590452353602874713527


/*
 * Structures
 */

// Parameters for transfer characteristics (gamma curves)
struct transfer {
	// Exponent used to linearize the signal
	float power;

	// Offset from 0.0 for the exponential curve
	float off;

	// Slope of linear segment near 0
	float slope;

	// Values below this are divided by slope during linearization
	float cutoffToLinear;

	// Values below this are multiplied by slope during gamma correction
	float cutoffToGamma;
};

// Parameters for a colorspace
struct rgb_space {
	// Chromaticity coordinates (xyz) for Red, Green, and Blue primaries
	mat4 primaries;

	// Chromaticity coordinates (xyz) for white point
	vec4 white;

	// Linearization and gamma correction parameters
	transfer trc;
};


/*
 * Preprocessor 'functions' that help build colorspaces as constants
 */

// Turns 6 chromaticity coordinates into a 3x3 matrix
#define Primaries(r1, r2, g1, g2, b1, b2)\
	mat4(\
		(r1), (r2), 1.0 - (r1) - (r2), 0,\
		(g1), (g2), 1.0 - (g1) - (g2), 0,\
		(b1), (b2), 1.0 - (b1) - (b2), 0,\
		0, 0, 0, 1)

// Creates a whitepoint's xyz chromaticity coordinates from the given xy coordinates
#define White(x, y)\
	vec4(vec3((x), (y), 1.0 - (x) - (y))/(y), 1)

// Automatically calculate the slope and cutoffs for transfer characteristics
#define Transfer(po, of)\
transfer(\
	(po),\
	(of),\
	(pow((po)*(of)/((po) - 1.0), 1.0 - (po))*pow(1.0 + (of), (po)))/(po),\
	(of)/((po) - 1.0),\
	(of)/((po) - 1.0)*(po)/(pow((po)*(of)/((po) - 1.0), 1.0 - (po))*pow(1.0 + (of), (po)))\
)

// Creates a scaling matrix using a vec4 to set the xyzw scalars
#define diag(v)\
	mat4(\
		(v).x, 0, 0, 0,\
		0, (v).y, 0, 0,\
		0, 0, (v).z, 0,\
		0, 0, 0, (v).w)

// Creates a conversion matrix that turns RGB colors into XYZ colors
#define rgbToXyz(space)\
	(space.primaries*diag(inverse((space).primaries)*(space).white))

// Creates a conversion matrix that turns XYZ colors into RGB colors
#define xyzToRgb(space)\
	inverse(rgbToXyz(space))


/*
 * Chromaticities for RGB primaries
 */

// CIE 1931 RGB
const mat4 primariesCie = Primaries(
	0.72329, 0.27671,
	0.28557, 0.71045,
	0.15235, 0.02
);

// Identity RGB
const mat4 primariesIdentity = mat4(1.0);



// Never-popular, antiquated, and idealized 'HDTV' primaries based mostly on the
// 1953 NTSC colorspace. SMPTE-240M officially used the SMPTE-C primaries
const mat4 primaries240m = Primaries(
	0.67, 0.33,
	0.21, 0.71,
	0.15, 0.06
);

// Alleged primaries for old Sony TVs with a very blue whitepoint
const mat4 primariesSony = Primaries(
	0.625, 0.34,
	0.28, 0.595,
	0.155, 0.07
);

// Rec. 709 (HDTV) and sRGB primaries
const mat4 primaries709 = Primaries(
	0.64, 0.33,
	0.3, 0.6,
	0.15, 0.06
);

// DCI-P3 primaries
const mat4 primariesDciP3 = Primaries(
	0.68, 0.32,
	0.265, 0.69,
	0.15, 0.06
);

// Rec. 2020 UHDTV primaries
const mat4 primaries2020 = Primaries(
	0.708, 0.292,
	0.17, 0.797,
	0.131, 0.046
);

// If the HUNT XYZ->LMS matrix were expressed instead as
// chromaticity coordinates, these would be them
const mat4 primariesHunt = Primaries(
	0.8374, 0.1626,
	2.3, -1.3,
	0.168, 0.0
);

// If the CIECAM97_1 XYZ->LMS matrix were expressed instead as
// chromaticity coordinates, these would be them
const mat4 primariesCiecam971 = Primaries(
	0.7, 0.306,
	-0.357, 1.26,
	0.136, 0.042
);

// If the CIECAM97_2 XYZ->LMS matrix were expressed instead as
// chromaticity coordinates, these would be them
const mat4 primariesCiecam972 = Primaries(
	0.693, 0.316,
	-0.56, 1.472,
	0.15, 0.067
);

// If the CIECAM02 XYZ->LMS matrix were expressed instead as
// chromaticity coordinates, these would be them
const mat4 primariesCiecam02 = Primaries(
	0.711, 0.295,
	-1.476, 2.506,
	0.144, 0.057
);

// LMS primaries as chromaticity coordinates, computed from
// http://www.cvrl.org/ciepr8dp.htm, and
// http://www.cvrl.org/database/text/cienewxyz/cie2012xyz2.htm
/*const mat3 primariesLms = Primaries(
	0.73840145, 0.26159855,
	1.32671635, -0.32671635,
	0.15861916, 0.0
);*/

// Same as above, but in fractional form
const mat4 primariesLms = Primaries(
	194735469.0/263725741.0, 68990272.0/263725741.0,
	141445123.0/106612934.0, -34832189.0/106612934.0,
	36476327.0/229961670.0, 0.0
);


/*
 * Chromaticities for white points
 */

// Standard Illuminant C. White point for the original 1953 NTSC color system
const vec4 whiteC = White(0.310063, 0.316158);

// Standard illuminant E (also known as the 'equal energy' white point)
const vec4 whiteE = White(1.0/3.0, 1.0/3.0);

// Alleged whitepoint to use with the P22 phosphors (D65 might be more proper)
const vec4 whiteP22 = White(0.313, 0.329);

// Standard illuminant D65. Note that there are more digits here than specified
// in either sRGB or Rec 709, so in some cases results may differ from other
// software. Color temperature is roughly 6504 K (originally 6500K, but complex
// science stuff made them realize that was innaccurate)
const vec4 whiteD65 = White(0.312713, 0.329016);

// Standard Illuminant D65 according to the Rec. 709 and sRGB standards
const vec4 whiteD65S = White(0.3127, 0.3290);

// Standard illuminant D50. Just included for the sake of including it. Content
// for Rec. 709 and sRGB is recommended to be produced using a D50 whitepoint.
// For the same reason as D65, the color temperature is 5003 K instead of 5000 K
const vec4 whiteD50 = White(0.34567, 0.35850);

// Standard Illuminant D50 according to ICC  specs
const vec4 whiteD50I = White(0.3457, 0.3585);

// White point for DCI-P3 Theater
const vec4 whiteTheater = White(0.314, 0.351);

// Very blue white point for old Sony televisions. Color temperature of 9300 K.
// Use with the 'primariesSony' RGB primaries defined above
const vec4 whiteSony = White(0.283, 0.298);


/*
 * Gamma curve parameters
 */

// Linear gamma
const transfer gam10 = transfer(1.0, 0.0, 1.0, 0.0, 0.0);

// Gamma of 1.8; not linear near 0. This is what older Apple devices used
const transfer gam18 = transfer(1.8, 0.0, 1.0, 0.0, 0.0);

// Gamma of 2.2; not linear near 0. Was defined abstractly to be used by early
// NTSC systems, before SMPTE 170M was modified to specify a more exact curve.
// Also what Adobe RGB uses
const transfer gam22 = transfer(2.2, 0.0, 1.0, 0.0, 0.0);

// Gamma of 2.4; not linear near 0. Used as the gamma value for BT.1886
const transfer gam24 = transfer(2.4, 0.0, 1.0, 0.0, 0.0);

// Gamma of 2.5; not linear near 0. Approximately what old Sony TVs used
const transfer gam25 = transfer(2.5, 0.0, 1.0, 0.0, 0.0);

// Gamma of 2.8; not linear near 0. Loosely defined gamma for European SDTV
const transfer gam28 = transfer(2.8, 0.0, 1.0, 0.0, 0.0);

// Modern SMPTE 170M, as well as Rec. 601, Rec. 709, and a rough approximation
// for Rec. 2020 content as well. Do not use with Rec. 2020 if you work with
// high bit depths!
const transfer gam170m = transfer(1.0/0.45, 0.099, 4.5, 0.0812, 0.018);

// Proper Rec. 2020 gamma, made using the new Transfer macro
const transfer gam2020 = Transfer(1.0/0.45, 0.099);

// Gamma for sRGB
const transfer gamSrgb = transfer(2.4, 0.055, 12.92, 0.04045, 0.0031308);

// A more continuous version of sRGB, for high bit depths
const transfer gamSrgbHigh = Transfer(2.4, 0.055);

// Gamma for the CIE L*a*b* Lightness scale
const transfer gamLab = transfer(3.0, 0.16, 243.89/27.0, 0.08, 216.0/24389.0);


/*
 * RGB Colorspaces
 */

// CIE 1931 RGB
const rgb_space Cie1931 = rgb_space(primariesCie, whiteE, gam10);

// Identity RGB
const rgb_space Identity = rgb_space(primariesIdentity, whiteE, gam10);


// Old Sony displays using high temperature white point
const rgb_space Sony = rgb_space(primariesSony, whiteSony, gam25);

// Rec. 709 (HDTV)
const rgb_space Rec709 = rgb_space(primaries709, whiteD65S, gam170m);

// sRGB (mostly the same as Rec. 709, but different gamma and full range values)
const rgb_space Srgb = rgb_space(primaries709, whiteD65S, gamSrgb);

// DCI-P3 D65
const rgb_space DciP3D65 = rgb_space(primariesDciP3, whiteD65S, gam170m);

// DCI-P3 D65
const rgb_space DciP3Theater = rgb_space(primariesDciP3, whiteTheater, gam170m);

// Rec. 2020
const rgb_space Rec2020 = rgb_space(primaries2020, whiteD65S, gam170m);

// Hunt primaries, balanced against equal energy white point
const rgb_space HuntRgb = rgb_space(primariesHunt, whiteE, gam10);

// CIE CAM 1997 primaries, balanced against equal energy white point
const rgb_space Ciecam971Rgb = rgb_space(primariesCiecam971, whiteE, gam10);

// CIE CAM 1997 primaries, balanced against equal energy white point and linearized
const rgb_space Ciecam972Rgb = rgb_space(primariesCiecam972, whiteE, gam10);

// CIE CAM 2002 primaries, balanced against equal energy white point
const rgb_space Ciecam02Rgb = rgb_space(primariesCiecam02, whiteE, gam10);

// Lms primaries, balanced against equal energy white point
const rgb_space LmsRgb = rgb_space(primariesLms, whiteE, gam10);


/*
 * Colorspace conversion functions
 */

// Converts RGB colors to a linear light scale
vec4 toLinear(vec4 color, const transfer trc)
{
	bvec4 cutoff = lessThan(color, vec4(trc.cutoffToLinear));
	bvec4 negCutoff = lessThanEqual(color, vec4(-1.0*trc.cutoffToLinear));
	vec4 higher = pow((color + trc.off)/(1.0 + trc.off), vec4(trc.power));
	vec4 lower = color/trc.slope;
	vec4 neg = -1.0*pow((color - trc.off)/(-1.0 - trc.off), vec4(trc.power));

	vec4 result = mix(higher, lower, cutoff);
	return mix(result, neg, negCutoff);
}

// Gamma-corrects RGB colors to be sent to a display
vec4 toGamma(vec4 color, const transfer trc)
{
	bvec4 cutoff = lessThan(color, vec4(trc.cutoffToGamma));
	bvec4 negCutoff = lessThanEqual(color, vec4(-1.0*trc.cutoffToGamma));
	vec4 higher = (1.0 + trc.off)*pow(color, vec4(1.0/trc.power)) - trc.off;
	vec4 lower = color*trc.slope;
	vec4 neg = (-1.0 - trc.off)*pow(-1.0*color, vec4(1.0/trc.power)) + trc.off;

	vec4 result = mix(higher, lower, cutoff);
	return mix(result, neg, negCutoff);
}

// Calculate Standard Illuminant Series D light source XYZ values
vec4 temperatureToXyz(float temperature)
{
	// Calculate terms to be added up. Since only the coefficients aren't
	// known ahead of time, they're the only thing determined by mix()
	float x = dot(mix(
		vec4(0.244063, 99.11, 2967800.0, -4607000000.0),
		vec4(0.23704, 247.48, 1901800.0, -2006400000.0),
		bvec4(temperature > 7000.0)
	)/vec4(1, temperature, pow(temperature, 2.0), pow(temperature, 3.0)), vec4(1));

	return White(x, -3.0*pow(x, 2.0) + 2.87*x - 0.275);
}


/*
 * Settings
 */

// Minimum temperature in the range (defined in the standard as 4000)
const float minTemp = 4000.0;

// Maximum temperature in the range (defined in the standard as 25000)
const float maxTemp = 9000.0;

// Display colorspace
const rgb_space display = Srgb;

// XYZ conversion matrices for the display colorspace
const mat4 toXyz = rgbToXyz(display);
const mat4 toRgb = xyzToRgb(display);

// LMS conversion matrices for white point adaptation
const mat4 toLms = xyzToRgb(LmsRgb);
const mat4 frLms = rgbToXyz(LmsRgb);







float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}




void getNightDesaturation(inout vec3 color, float lmx) {
	float lum = dot(color,vec3(0.15,0.3,0.55));
	float lum2 = dot(color,vec3(0.85,0.7,0.45))/2;
	float rodLum = lum2*300.0;
	float rodCurve = mix(1.0, rodLum/(2.5+rodLum), 1*(Purkinje_strength));
	color = mix(lum*lmx*vec3(Purkinje_R, Purkinje_G, Purkinje_B), color, rodCurve);

	float brightness = dot(color, vec3(0.2627, 0.6780, 0.0593));
	float amount = clamp(0.1 / (pow(brightness * ndeSat, 2.0) + 0.02),0,1);
	vec3 desatColor = mix(color, vec3(brightness), vec3(0.9)) * vec3(0.2, 1.0, 2.0);

	color = mix(color, desatColor, amount);


}

float interleaved_gradientNoise(){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y));
}
vec3 int8Dither(vec3 color){
	float dither = interleaved_gradientNoise();
	return color + dither*exp2(-8.0);
}

void BSLTonemap(inout vec3 color){
	color = EXPOSURE * color;
	color = color / pow(pow(color, vec3(TONEMAP_WHITE_CURVE)) + 1.0, vec3(1.0 / TONEMAP_WHITE_CURVE));
	color = pow(color, mix(vec3(TONEMAP_LOWER_CURVE), vec3(TONEMAP_UPPER_CURVE), sqrt(color)));
}
vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}


float cdist(vec2 coord) {
	return max(abs(coord.s-0.5),abs(coord.t-0.5))*2.0;
}
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
void main() {

    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    vec2 oneTexel = 1/ScreenSize;
    vec2 lmtrans = unpackUnorm2x4((texture(MainSampler, texCoord).a));
    vec2 lmtrans3 = unpackUnorm2x4((texture(MainSampler, texCoord+oneTexel.y).a));

          

    float lmy = mix(lmtrans.y,lmtrans3.y,res);
    float lmx = mix(lmtrans3.y,lmtrans.y,res);

    float depth = texture(DiffuseDepthSampler, texCoord).r;


    vec3 color = texture(DiffuseSampler, texCoord).rgb;


    vec2 uv = gl_FragCoord.xy / ScreenSize.xy/2. +.25;

    float vignette = (1.5-dot(texCoord-0.5,texCoord-0.5)*2.);

    float i = SAMPLE_OFFSET;
    i = i * sin(1 * 0.5 + vec3(0, 0, 0)).x; // make this animated
    
    vec3 img = texture( DiffuseSampler, uv*2.-.5).rgb;
      
    vec3 col = texture( blursampler, uv + vec2( i, i ) / ScreenSize ).rgb / 6.0;
  
    col += texture( blursampler, uv + vec2( i, -i ) / ScreenSize ).rgb / 6.0;
    col += texture( blursampler, uv + vec2( -i, i ) / ScreenSize ).rgb / 6.0;
    col += texture( blursampler, uv + vec2( -i, -i ) / ScreenSize ).rgb / 6.0;
    
    col += texture( blursampler, uv + vec2( 0    , i*2.0 ) / ScreenSize ).rgb / 12.0;
    col += texture( blursampler, uv + vec2( i*2. , 0     ) / ScreenSize ).rgb / 12.0;
    col += texture( blursampler, uv + vec2( -i*2., 0     ) / ScreenSize ).rgb / 12.0;
    col += texture( blursampler, uv + vec2( 0    , -i*2. ) / ScreenSize ).rgb / 12.0;
         col *= col;
    vec3 fin = max(vec3(0.0), col - 0.03);
    

	float lightScat = clamp(5.0*0.05*pow(1,0.2),0.0,1.0)*vignette;

    float VL_abs =  texture(BloomSampler, texCoord).a;
	float purkinje = 1/(1.0+1)*Purkinje_strength;
    VL_abs = clamp((1.0-VL_abs)*1.0*0.75*(1.0-purkinje),0.0,1.0)*clamp(1.0-pow(cdist(texCoord.xy),15.0),0.0,1.0);
	color = (mix(color*1.5,col,VL_abs)+fin*lightScat);
//         lmx *= clamp(pow(depth,512)*10,0,1);
	getNightDesaturation(color.rgb,clamp((lmx+lmy),0.0,5));	


	// Calculate color temperature for pixel
	vec2 uv2 = texCoord;
	float temp = uv2.x*(maxTemp - minTemp) + minTemp;

	// Calculate temperature conversion
	float selected = (5600.0 - minTemp)/(maxTemp - minTemp);

	vec4 convWhite = temperatureToXyz(selected*(maxTemp - minTemp) + minTemp);
	mat4 adapt = toRgb*frLms*diag((toLms*convWhite)/(toLms*display.white))*toLms*toXyz;
	color = mat3(adapt)*color;

	BSLTonemap(color);
    float lumC = luma(color);
	vec3 diff = color-lumC;
	color = color + diff*(-lumC*CROSSTALK + SATURATION);
  //  color.rgb = vec3(VL_abs);









	fragColor= vec4(int8Dither(vec3(color)), 1.0);
    
}
