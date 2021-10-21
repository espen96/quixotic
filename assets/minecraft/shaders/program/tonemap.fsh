#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D MainSampler;
uniform sampler2D BloomSampler;
uniform sampler2D blursampler;
uniform vec2 ScreenSize;
uniform vec2 OutSize;
out vec4 fragColor;

in vec2 texCoord;




    #define EXPOSURE 1.30 
    #define TONEMAP_WHITE_CURVE 1.7 
    #define TONEMAP_LOWER_CURVE 1.2 
    #define TONEMAP_UPPER_CURVE 1.3 
    #define CROSSTALK 0.15 // Desaturates bright colors and preserves saturation in darker areas (inverted if negative). Helps avoiding almsost fluorescent colors 
    #define SATURATION 0.25 // Negative values desaturates colors, Positive values saturates color, 0 is no change
    
    #define ndeSat 7.0
    #define Purkinje_strength 1.0	// Simulates how the eye is unable to see colors at low light intensities. 0 = No purkinje effect at low exposures 
    #define Purkinje_R 0.4
    #define Purkinje_G 0.7 
    #define Purkinje_B 1.0
    #define Purkinje_Multiplier 0.1 // How much the purkinje effect increases brightness


    #define SAMPLE_OFFSET 5.
    #define INTENSITY 0.1


float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}


vec4 textureGood( sampler2D sam, vec2 uv )
{
    vec2 res = textureSize( sam,0 );

    vec2 st = uv*res - 0.5;

    vec2 iuv = floor( st );
    vec2 fuv = fract( st );

    vec4 a = textureLod( sam, (iuv+vec2(0.5,0.5))/res ,0);
    vec4 b = textureLod( sam, (iuv+vec2(1.5,0.5))/res ,0);
    vec4 c = textureLod( sam, (iuv+vec2(0.5,1.5))/res ,0);
    vec4 d = textureLod( sam, (iuv+vec2(1.5,1.5))/res ,0);

    return mix( mix( a, b, fuv.x),
                mix( c, d, fuv.x), fuv.y );
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
#define NUMCONTROLS 100
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define FUDGE 32.0
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
    vec2 oneTexel = 1/ScreenSize;
    if (inctrl) {
        return (texture(inSampler, coords - vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(0.0, oneTexel.y))) / 3.0;
    } else {
        return texture(inSampler, coords);
    }
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

    bool inctrl = inControl(texCoord * OutSize, OutSize.x) > -1;
    vec3 color = texture(DiffuseSampler, texCoord).rgb;



    vec2 uv = gl_FragCoord.xy / ScreenSize.xy/2. +.25;

    float vignette = (1.5-dot(texCoord-0.5,texCoord-0.5)*2.);

    float i = SAMPLE_OFFSET;
    i = i * sin(1 * 0.5 + vec3(0, 0, 0)).x; 
    
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
    vec3 fin = col;
    

	float lightScat = clamp(5.0*0.05*pow(1,0.2),0.0,1.0)*vignette;

    float VL_abs =  texture(BloomSampler, texCoord).a;
	float purkinje = 1/(1.0+1)*Purkinje_strength;
    VL_abs = clamp((1.0-VL_abs)*1.0*0.75*(1.0-purkinje),0.0,1.0)*clamp(1.0-pow(cdist(texCoord.xy),15.0),0.0,1.0);
	color = (mix(color*1.5,col,VL_abs)+fin*lightScat);
	getNightDesaturation(color.rgb,clamp((lmx+lmy),0.0,5));	
    /*
    vec3 color_pick = vec3(162,203,221)/255;
    vec2 xyEst = XYZ2xy(sRGBtoXYZ*color_pick);
    vec3 xyzEst = xy2XYZ(xyEst,100.0);
    mat3 M = cbCAT(xyzEst, xyz_D65);
    color = M*color;
    */
    	BSLTonemap(color);
    float lumC = luma(color);
	vec3 diff = color-lumC;
	color = color + diff*(-lumC*CROSSTALK + SATURATION);
  //  color.rgb = vec3(VL_abs);

	fragColor= vec4(int8Dither(vec3(color.rgb)), 1.0);
    
}
