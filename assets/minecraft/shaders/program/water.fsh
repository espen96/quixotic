#version 150
out vec4 fragColor;

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TerrainCloudsSampler;
uniform sampler2D temporals3Sampler;
uniform vec2 OutSize;
uniform vec2 ScreenSize;
in vec2 texCoord;
in vec2 oneTexel;
in float aspectRatio;
in float cosFOVrad;
in float tanFOVrad;
in mat4 gbPI;
in mat4 gbP;
flat in vec3 ambientUp;
flat in vec3 ambientLeft;
flat in vec3 ambientRight;
flat in vec3 ambientB;
flat in vec3 ambientF;
flat in vec3 ambientDown;
flat in vec3 avgSky;
flat in float isEyeInLava;
flat in float isEyeInWater;
#define near 0.00004882812 
in float far;

#define NORMDEPTHTOLERANCE 1.0
#define SSR_TAPS 3
#define SSR_SAMPLES 10
#define SSR_MAXREFINESAMPLES 10
#define SSR_STEPREFINE 0.2
#define SSR_STEPINCREASE 1.2
#define SSR_IGNORETHRESH 0.0
#define NORMAL_SCATTER 0.006








in vec2 texCoord2;

in vec3 sunDir;
in vec4 fogcol;
in mat4 gbufferModelViewInverse;
in mat4 gbufferModelView;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;

//Thanks to Jessie for dithering
float Bayer2  (vec2 c) { c = 0.5 * floor(c); return fract(1.5 * fract(c.y) + c.x); }
float Bayer4  (vec2 c) { return 0.25 * Bayer2  (0.5 * c) + Bayer2(c); }
float Bayer8  (vec2 c) { return 0.25 * Bayer4  (0.5 * c) + Bayer2(c); }
float Bayer16 (vec2 c) { return 0.25 * Bayer8  (0.5 * c) + Bayer2(c); }
float Bayer32 (vec2 c) { return 0.25 * Bayer16 (0.5 * c) + Bayer2(c); }
float Bayer64 (vec2 c) { return 0.25 * Bayer32 (0.5 * c) + Bayer2(c); }
float Bayer128(vec2 c) { return 0.25 * Bayer64 (0.5 * c) + Bayer2(c); }
float Bayer256(vec2 c) { return 0.25 * Bayer128(0.5 * c) + Bayer2(c); }
  
float LinearizeDepth(float depth) 
{
    return (2.0 * near * far) / (far + near - depth * (far - near));    
}

float ditherGradNoise() {
  return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}

float luminance(vec3 rgb) {
    float redness = clamp(dot(rgb, vec3(1.0, -0.25, -0.75)), 0.0, 1.0);
    return ((1.0 - redness) * dot(rgb, vec3(0.2126, 0.7152, 0.0722)) + redness * 1.4) * 4.0;
}

float luma4(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}



vec4 SSR(vec3 fragpos, float fragdepth, vec3 surfacenorm, vec4 skycol, vec4 approxreflection, vec2 randsamples[64]) {
    vec3 rayStart   = fragpos.xyz;
    vec3 rayDir     = reflect(normalize(fragpos.xyz), surfacenorm);
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
        vec3 colortmp = texture(TerrainCloudsSampler, pos.xy).rgb*2.0;
        candidate = mix(vec4(colortmp, 1.0), skycol, float(dtmp + SSR_IGNORETHRESH < 1.0) * clamp(pos.z * 1.1, 0.0, 1.0));
    }
    
    candidate = mix(candidate, skycol, pos.y );

    return candidate;
}



vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

////////////////////////////



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
vec4 getNotControl(sampler2D inSampler, vec2 coords, bool inctrl) {
    if (inctrl) {
        return (texture(inSampler, coords - vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(oneTexel.x, 0.0)) + texture(inSampler, coords + vec2(0.0, oneTexel.y))) / 3.0;
    } else {
        return texture(inSampler, coords);
    }
}

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define FUDGE 32.0

vec4 backProject(vec4 vec) {
    vec4 tmp = gbufferModelViewInverse * vec;
    return tmp / tmp.w;
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

vec3 lumaBasedReinhardToneMapping(vec3 color)
{
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma / (1. + luma);
	color *= toneMappedLuma / luma;
	color = pow(color, vec3(1. / 2.2));
	return color;
}



void main() {


    vec3 reflection = vec3(1.0);
    

    vec4 color = texture(TranslucentSampler, texCoord);


    vec4 color2 = color;

    if (color.a > 0.01 ) {
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    vec2 lmtrans = unpackUnorm2x4((texture(DiffuseSampler, texCoord).a));
    vec2 lmtrans3 = unpackUnorm2x4((texture(DiffuseSampler, texCoord+oneTexel.y).a));

    float lmx = 0;
    float lmy = 0;


          lmy = mix(lmtrans.y,lmtrans3.y,res);
          lmx = clamp(mix(lmtrans3.y,lmtrans.y,res)*2.0,0,1);
    float lm = clamp(lmx-lmy,0,1);


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

    vec3 sky = mix(color.rgb*2.0,avgSky,0.5);

    float wdepth = texture(TranslucentDepthSampler, texCoord).r;
    float wdepth2 = texture(TranslucentDepthSampler, texCoord + vec2(0.0, oneTexel.y)).r;
    float wdepth3 = texture(TranslucentDepthSampler, texCoord + vec2(oneTexel.x, 0.0)).r;
    float ldepth = LinearizeDepth(wdepth);
    float ldepth2 = LinearizeDepth(wdepth2);
    float ldepth3 = LinearizeDepth(wdepth3);
    ldepth2 = abs(ldepth - ldepth2) > NORMDEPTHTOLERANCE ? ldepth : ldepth2;
    ldepth3 = abs(ldepth - ldepth3) > NORMDEPTHTOLERANCE ? ldepth : ldepth3;




    bool inctrl = inControl(texCoord * OutSize, OutSize.x) > -1;
        vec3 fragpos = (gbPI * vec4(texCoord, ldepth, 1.0)).xyz;
        fragpos *= ldepth;
        vec3 p8 = (gbPI * vec4(texCoord + vec2(0.0, oneTexel.y), ldepth2, 1.0)).xyz;
        p8 *= ldepth2;
        vec3 p7 = (gbPI * vec4(texCoord + vec2(oneTexel.x, 0.0), ldepth3, 1.0)).xyz;
        p7 *= ldepth3;
        vec3 normal = -normalize(cross(p8 - fragpos, p7 - fragpos));
        
        float ndlsq = dot(normal, vec3(0.0, 0.0, 1.0));
                float horizon = clamp(ndlsq * 100000.0, -1.0, 1.0);

        ndlsq = ndlsq * ndlsq;
    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);

        // first calculate approximate surface normal using depth map
        float depth2 = getNotControl(TranslucentDepthSampler, texCoord + vec2(0.0, oneTexel.y), inctrl).r;
        float depth3 = getNotControl(TranslucentDepthSampler, texCoord + vec2(oneTexel.x, 0.0), inctrl).r;
        float depth4 = getNotControl(TranslucentDepthSampler, texCoord - vec2(0.0, oneTexel.y), inctrl).r;
        float depth5 = getNotControl(TranslucentDepthSampler, texCoord - vec2(oneTexel.x, 0.0), inctrl).r;
    float depth = texture(TranslucentDepthSampler, texCoord).r;

   vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));
     vec3 fragpos2 = backProject(vec4(scaledCoord, depth, 1.0)).xyz;

        vec3 p2 = backProject(vec4(scaledCoord + 2.0 * vec2(0.0, oneTexel.y), depth2, 1.0)).xyz;
        p2 = p2 - fragpos2;
        vec3 p3 = backProject(vec4(scaledCoord + 2.0 * vec2(oneTexel.x, 0.0), depth3, 1.0)).xyz;
        p3 = p3 - fragpos2;


        vec4 r = vec4(0.0);
        for (int i = 0; i < SSR_TAPS; i += 1) {
            r += SSR(fragpos, ldepth, normalize(normal + NORMAL_SCATTER * (normalize(p2) * poissonDisk[i].x + normalize(p3) * poissonDisk[i].y)), vec4(sky,1), vec4(sky,1), poissonDisk);


            
        }
        reflection = r.rgb / SSR_TAPS;
        
        float fresnel = pow(1.0 - pow(dot(normalize(fragpos), normal), 0.8), 3.0);
              fresnel = clamp(exp((fresnel - 1.0) * (4.0 + clamp(exp(clamp(ndlsq - 0.05, 0.0, 1.0) * 2.0) - 1.0, 0.0, 1.0) * 25.0)), 0.0, 1.0);
              fresnel = clamp(exp(-35 * pow(dot(normalize(fragpos), normal), 2.0))*5.0,0,1);
        float lookfresnel = clamp(exp(-25 * clamp(ndlsq * horizon, 0.0, 1.0) + 3.0)*100, 0.0, 1.0);
	

           color = vec4(mix(sky*0.7,reflection,(fresnel *lookfresnel)), color.a);
           color = ((color2*luma4(sky.rgb) ) +color)*0.6;
           color = mix(color2,color,1-clamp(luminance(color2.rgb)*luminance(color2.rgb)*0.1,0,1));
           color = pow(color,vec4(1.2));

//         color = vec4(reflection.rgb,color.a);

    }        
   

    fragColor= vec4(color.rgb,color2.a);
}
