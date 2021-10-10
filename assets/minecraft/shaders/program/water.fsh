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
in mat4 wgbufferModelViewInverse;

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


  
float LinearizeDepth(float depth) 
{
    return (2.0 * near * far) / (far + near - depth * (far - near));    
}



float luminance(vec3 rgb) {
    float redness = clamp(dot(rgb, vec3(1.0, -0.25, -0.75)), 0.0, 1.0);
    return ((1.0 - redness) * dot(rgb, vec3(0.2126, 0.7152, 0.0722)) + redness * 1.4) * 4.0;
}

float luma4(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}

float ditherGradNoise() {
  return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}




vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + vec4(0,0,gbufferProjectionInverse[3].ba);
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

vec4 Raytrace(sampler2D depthtex, vec3 viewPos, vec3 normal, float dither, float fresnelRT) {
	vec3 pos = vec3(0.0);
	float dist = 0.0;

	#if AA > 1
		dither = fract(dither + frameTimeCounter);
	#endif

	vec3 start = viewPos;
	vec3 nViewPos = normalize(viewPos);
    vec3 vector = 0.5 * reflect(nViewPos, normalize(normal));
    viewPos += vector;
	vec3 tvector = vector;

	float difFactor = fresnelRT;

    int sr = 0;

    for(int i = 0; i < 15; i++) {
        pos = nvec3(gbufferProjection * nvec4(viewPos)) * 0.5 + 0.5;
		if (pos.x < -0.05 || pos.x > 1.05 || pos.y < -0.05 || pos.y > 1.05) break;

		vec3 rfragpos = vec3(pos.xy, texture2D(depthtex,pos.xy).r);
        rfragpos = nvec3(gbufferProjectionInverse * nvec4(rfragpos * 2.0 - 1.0));
		dist = length(start - rfragpos);

        float err = length(viewPos - rfragpos);
		float lVector = length(vector);
		float dif = length(start - rfragpos);
		if (err < pow(lVector, 1.14) || (dif < difFactor && err > difFactor)) {
                sr++;
                if(sr >= 6) break;
				tvector -= vector;
                vector *= 0.1;
		}
        vector *= 2.0;
        tvector += vector * (dither * 0.05 + 0.75);
		viewPos = start + tvector;
    }

	return vec4(pos, dist);
}
vec4 SSR(vec3 fragpos, float fragdepth, vec3 surfacenorm, vec4 skycol, float fresnel) {

    vec3 pos    = vec3(0.0);




    vec4 color = vec4(0.0);

     pos = Raytrace(DiffuseDepthSampler, fragpos, surfacenorm, ditherGradNoise(), fresnel).xyz;

	float border = clamp(1.0 - pow(cdist(pos.st), 50.0), 0.0, 1.0);
	
	if (pos.z < 1.0 - 1e-5) {
		float refDepth = texture2D(TranslucentDepthSampler, pos.st).r;
		color.a = float(0.999999 > refDepth);
		if (color.a > 0.001) {
			color.rgb = texture2D(TerrainCloudsSampler, pos.st).rgb*border;
			if (refDepth > 0.9995) color.rgb *= 1;
		}
		color.a *= border;
	}

//	color.rgb = pow(color.rgb * 2.0, vec3(8.0));

    return color;
}



vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

////////////////////////////

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define FUDGE 32.0


vec4 backProject(vec4 vec) {
    vec4 tmp = wgbufferModelViewInverse * vec;
    return tmp / tmp.w;
}
vec3 worldToView(vec3 worldPos) {

    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos +gbufferModelView[3];

    return pos.xyz;
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


    vec2 poissonDisk[16];
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


    vec3 sky = mix(color.rgb*2.0,avgSky,0.5);

    float deptha = texture(TranslucentDepthSampler, texCoord).r;
    float depthb = texture(TranslucentDepthSampler, texCoord-vec2(0,oneTexel.y)).r;
    float depthc = texture(TranslucentDepthSampler, texCoord+vec2(0,oneTexel.y)).r;
    float depthd = texture(TranslucentDepthSampler, texCoord+vec2(oneTexel.x,0)).r;
    float depthe = texture(TranslucentDepthSampler, texCoord-vec2(oneTexel.x,0)).r;

    vec2 scaledCoord = 2.0 * (texCoord - vec2(0.5));

        float depth2 = depthc;
        float depth3 = depthd;
        float depth4 = depthb;
        float depth5 = depthe;

        vec3 fragpos = backProject(vec4(scaledCoord, deptha, 1.0)).xyz;

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


      vec3 normal3 = worldToView (normal);




    float wdepth = texture(TranslucentDepthSampler, texCoord).r;
    float wdepth2 = texture(TranslucentDepthSampler, texCoord + vec2(0.0, oneTexel.y)).r;
    float wdepth3 = texture(TranslucentDepthSampler, texCoord + vec2(oneTexel.x, 0.0)).r;
    float ldepth = LinearizeDepth(wdepth);
    float ldepth2 = LinearizeDepth(wdepth2);
    float ldepth3 = LinearizeDepth(wdepth3);
    ldepth2 = abs(ldepth - ldepth2) > NORMDEPTHTOLERANCE ? ldepth : ldepth2;
    ldepth3 = abs(ldepth - ldepth3) > NORMDEPTHTOLERANCE ? ldepth : ldepth3;



        vec3 fragpos2 = (gbPI * vec4(texCoord, ldepth, 1.0)).xyz;
        fragpos *= ldepth;
        vec3 p8 = (gbPI * vec4(texCoord + vec2(0.0, oneTexel.y), ldepth2, 1.0)).xyz;
        p8 *= ldepth2;
        vec3 p7 = (gbPI * vec4(texCoord + vec2(oneTexel.x, 0.0), ldepth3, 1.0)).xyz;
        p7 *= ldepth3;
        vec3 normal2 = -normalize(cross(p8 - fragpos, p7 - fragpos));
        
        float ndlsq = dot(normal, vec3(0.0, 0.0, 1.0));
                float horizon = clamp(ndlsq * 100000.0, -1.0, 1.0);

        ndlsq = ndlsq * ndlsq;
    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);
    float z = texture(TranslucentDepthSampler,texCoord).x;
    vec3 screenPos2 = vec3(texCoord, z);
    vec3 clipPos = screenPos2 * 2.0 - 1.0;
    vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
    vec3 viewPos = tmp.xyz / 1.0;	

            vec3 fragpos3 = (1 * vec4(texCoord, ldepth, 1.0)).xyz;
            fragpos3 *= ldepth;
            float fresnel = pow(clamp(1.0 + dot(normal3, normalize(fragpos3.xyz)), 0.0, 1.0), 5.0);



        vec4 r = vec4(0.0);
        for (int i = 0; i < SSR_TAPS; i += 1) {
            r += SSR(viewPos, ldepth, normalize(normal3 + NORMAL_SCATTER * (normalize(p2) * poissonDisk[i].x + normalize(p3) * poissonDisk[i].y)), vec4(sky,1),fresnel);
   
        }
        reflection = r.rgb / SSR_TAPS;
        
		
           color = vec4(mix(sky,reflection,r.a/SSR_TAPS), color.a);
           color = ((color2*luma4(sky.rgb) ) +color)*0.6;
           color = mix(color2,color,1-clamp(luminance(color2.rgb)*luminance(color2.rgb)*0.1,0,1));
           color = pow(color,vec4(1.2));

  //      color = vec4(viewPos.rgb,color.a);

    }        
   

    fragColor= vec4(color.rgb,color2.a);
}
