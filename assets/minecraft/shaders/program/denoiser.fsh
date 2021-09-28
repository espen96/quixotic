#version 150

in vec2 texCoord;
in vec2 oneTexel;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in mat4 gbufferModelView;
in mat4 gbufferModelViewInverse;
in mat4 gbufferPreviousProjection;
in mat4 gbufferPreviousModelView;
in mat4 projInv;
in mat4 projInv2;
in vec3 currChunkOffset;
in vec3 prevChunkOffset;
in float near;
in float far;
in vec3 prevPosition;

uniform sampler2D DiffuseSampler;
uniform sampler2D CurrentFrameDepthSampler;
uniform vec2 ScreenSize;
uniform sampler2D PreviousFrameSampler;
uniform sampler2D PreviousFrameDepthSampler;

out vec4 fragColor;

#define BLEND_FACTOR 0.5 //[0.01 0.02 0.03 0.04 0.05 0.06 0.08 0.1 0.12 0.14 0.16] higher values = more flickering but sharper image, lower values = less flickering but the image will be blurrier
#define MOTION_REJECTION 0.5 //[0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5] //Higher values=sharper image in motion at the cost of flickering
#define ANTI_GHOSTING 0.5 //[0.0 0.25 0.5 0.75 1.0] High values reduce ghosting but may create flickering
#define FLICKER_REDUCTION 0.0  //[0.0 0.25 0.5 0.75 1.0] High values reduce flickering but may reduce sharpness


vec3 calculateWorldPos(float depth, vec2 texCoord, mat4 projMat, mat4 modelViewMat) {
    
    vec4 clip = vec4(texCoord * 2 - 1, depth, 1);
    vec4 viewSpace = inverse(projMat) * clip;
    viewSpace /= viewSpace.w;
    return (inverse(modelViewMat) * viewSpace).xyz;
}


	float MinDepth3x3(vec2 uv) {
		float minDepth = texture(CurrentFrameDepthSampler, uv).x;
		for (int x = -1; x <= 1; ++x) {
			for (int y = -1; y <= 1; ++y) {
				if (x == 0 && y == 0) { continue; }
				minDepth = min(minDepth, texture(CurrentFrameDepthSampler, uv + oneTexel * vec2(x, y)).x);
			}
		}
		return minDepth;
	}
vec3 ViewSpaceFromScreenSpace(vec3 pos, mat4 inverseProjection) {
	pos = pos * 2.0 - 1.0;
	#ifdef TAA
		pos.xy -= taaOffset;
	#endif

	vec3 viewPosition  = vec3(vec2(inverseProjection[0].x, inverseProjection[1].y) * pos.xy + inverseProjection[3].xy, inverseProjection[3].z);
	     viewPosition /= inverseProjection[2].w * pos.z + inverseProjection[3].w;

	return viewPosition;
}
vec3 ScreenSpaceFromViewSpace(vec3 viewPosition, mat4 projection) {
	vec3 screenPosition = vec3(projection[0].x, projection[1].y, projection[2].z) * viewPosition + projection[3].xyz;
	#ifdef TAA
		screenPosition.xy -= taaOffset * viewPosition.z;
	#endif
	return screenPosition * (0.5 / -viewPosition.z) + 0.5;
}

float luma(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}
	vec3 Reproject(vec3 position) {
		if (position.z >= 1.0) {
			position = ViewSpaceFromScreenSpace(position, gbufferProjectionInverse);
			position = mat3(gbufferModelViewInverse) * position;
			position = mat3(gbufferPreviousModelView) * position;
			position = ScreenSpaceFromViewSpace(position, gbufferPreviousProjection);
		} else if (position.z > 0.6) {
			position = ViewSpaceFromScreenSpace(position, gbufferProjectionInverse);
			position = mat3(gbufferModelViewInverse) * position + gbufferModelViewInverse[3].xyz;
			position = position + currChunkOffset - prevChunkOffset;
			position = mat3(gbufferPreviousModelView) * position + gbufferPreviousModelView[3].xyz;
			position = ScreenSpaceFromViewSpace(position, gbufferPreviousProjection);
		}

		return position;
	}
// From a presentation given by Lasse Jon Fuglsang Pedersen titled "Temporal Reprojection Anti-Aliasing in INSIDE"
// https://www.youtube.com/watch?v=2XXS5UyNjjU&t=434s
vec3 clipColor(vec3 aabbMin, vec3 aabbMax, vec3 prevColor) {
    // Center of the clip space
    vec3 pClip = (aabbMax + aabbMin) / 2;
    // Size of the clip space
    vec3 eClip = (aabbMax - aabbMin) / 2;

    // The relative coordinates of the previous color in the clip space
    vec3 vClip = prevColor - pClip;
    // Normalized clip space coordintes
    vec3 vUnit = vClip / eClip;
    // The distance of the previous color from the center of the clip space in each axis in the normalized clip space
    vec3 aUnit = abs(vUnit);
    // The divisor is the largest distance from the center along each axis
    float divisor = max(aUnit.x, max(aUnit.y, aUnit.z));
    if (divisor > 1) {
        // If the divisor is larger, than 1, that means that the previous color is outside of the clip space
        // If we divide by divisor, we'll put it into clip space
        return pClip + vClip / divisor;
    }
    // Otherwise it's already clipped
    return prevColor;
}


/*
    -- Vertex Engine X --

    Copyright 2020 UAA Software

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
    OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#define VXAA_TEXTURE_CURRENT DiffuseSampler
#define VXAA_TEXTURE_PREV PreviousFrameSampler

#define VXAA_TEMPORALEDGE_THRES 0.025
#define VXAA_TEMPORALEDGE_TIME_MIN 0.0000001
#define VXAA_TEMPORALEDGE_TIME_MAX 1.15
#define VXAA_SPATIAL_FLICKER_TIME 2.35
#define VXAA_MORPHOLOGICAL_STRENGTH 0.1
#define VXAA_MORPHOLOGICAL_SHARPEN 0.1
#define iTimeDelta 0.1
#define VXAA_W 0
#define VXAA_E 1
#define VXAA_N 2
#define VXAA_S 3
#define VXAA_NW 0
#define VXAA_NE 1
#define VXAA_SW 2
#define VXAA_SE 3

float saturate( float x )
{
    return clamp( x, 0.0, 1.0 );
}

vec4 pow3( vec4 x, float y )
{
    return vec4( pow( x.x, y ), pow( x.y, y ), pow( x.z, y ), x.w );
}

float VXAALuma( vec3 c )
{
    return dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
}

float VXAALumaDiff( vec3 x, vec3 y )
{
    float l1 = dot( x, vec3( 0.2126, 0.7152, 0.0722 ) );
    float l2 = dot( y, vec3( 0.2126, 0.7152, 0.0722 ) );
    return abs( l1 - l2 );
}

float VXAATemporalContrast( float currentLuma, float historyLuma )
{
    float x = saturate( abs( historyLuma - currentLuma ) - VXAA_TEMPORALEDGE_THRES );
    float x2 = x * x, x3 = x2 * x;
    return saturate( 3.082671957671837 * x - 3.9384920634917364 * x2 + 1.8518518518516354 * x3 );
}

float VXAAMorphStrengthShaper( float x )
{
    return 1.3 * x - 0.3 * x * x;
}

float VXAASpatialContrast( vec2 spatialLumaMinMax )
{
    float spatialContrast = spatialLumaMinMax.y - spatialLumaMinMax.x;
    return mix( 0.0f, 1.0f, spatialContrast );
}

float VXAATemporalFilterAlpha( float fpsRcp, float convergenceTime )
{
    return exp( -fpsRcp / convergenceTime );
}

vec3 VXAAClampHistory( vec3 history, vec4 currN[4] )
{
    vec3 cmin = min( min( currN[0].rgb, currN[1].rgb ), min( currN[2].rgb, currN[3].rgb ) );
    vec3 cmax = max( min( currN[0].rgb, currN[1].rgb ), max( currN[2].rgb, currN[3].rgb ) );
    return vec3(
        clamp( history.r, cmin.r, cmax.r ),
        clamp( history.g, cmin.g, cmax.g ),
        clamp( history.b, cmin.b, cmax.b )
    );
}

vec4 VXAASharpen( vec4 history, vec4 histN[4] )
{
    vec4 nh = histN[VXAA_NW] + histN[VXAA_NE] + histN[VXAA_SW] + histN[VXAA_SE];
    return mix( history, history * 5.0f - nh, VXAA_MORPHOLOGICAL_SHARPEN );
}

vec4 VXAAMorphological( vec2 uv, vec4 current, vec4 currN[4], float strength )
{
    if ( strength < 0.1f ) return current;
    float lumaNW = currN[VXAA_NW].a, lumaNE = currN[VXAA_NE].a,
        lumaSW = currN[VXAA_SW].a, lumaSE = currN[VXAA_SE].a;
    lumaNE += 0.0025;
    float lumaMin = min( current.a, min( min( lumaNW, lumaNE ), min( lumaSW, lumaSE ) ) );
    float lumaMax = max( current.a, max( max( lumaNW, lumaNE ), max( lumaSW, lumaSE ) ) );
    
    vec2 dir;
    dir.x = ( lumaSW - lumaNE ) + ( lumaSE - lumaNW );
    dir.y = ( lumaSW - lumaNE ) - ( lumaSE - lumaNW );
    vec2 dirN = normalize( dir );
    
    vec4 n1 = texture( VXAA_TEXTURE_CURRENT, uv - dirN * strength / ScreenSize.xy );
    vec4 p1 = texture( VXAA_TEXTURE_CURRENT, uv + dirN * strength / ScreenSize.xy );
    return ( n1 + p1 ) * 0.5;
}

vec4 VXAAFilmic( vec2 uv, vec4 current, vec4 history, vec4 currN[4], vec4 histN[4] )
{
    // Temporal contrast weight.
    float temporalContrastWeight = VXAATemporalContrast( current.a, history.a );

    // Spatial contrast weight.
    vec2 spatialLumaMinMaxC = vec2(
        min( min( currN[0].a, currN[1].a ), min( currN[2].a, currN[3].a ) ),
        max( max( currN[0].a, currN[1].a ), max( currN[2].a, currN[3].a ) )
    );
    vec2 spatialLumaMinMaxH = vec2(
        min( min( histN[0].a, histN[1].a ), min( histN[2].a, histN[3].a ) ),
        max( max( histN[0].a, histN[1].a ), max( histN[2].a, histN[3].a ) )
    );
    float spatialContrastWeightC = VXAASpatialContrast( spatialLumaMinMaxC );
    float spatialContrastWeightH = VXAASpatialContrast( spatialLumaMinMaxH );
    float spatialContrastWeight = abs( spatialContrastWeightC - spatialContrastWeightH );
    
    // Evaluate convergence time from weights.
    float convergenceTime = mix( VXAA_TEMPORALEDGE_TIME_MIN, VXAA_TEMPORALEDGE_TIME_MAX, temporalContrastWeight );
    convergenceTime = mix( convergenceTime, VXAA_SPATIAL_FLICKER_TIME, spatialContrastWeight );
    float alpha = VXAATemporalFilterAlpha( iTimeDelta, convergenceTime );

    
    // Apply morpholigical AA filter and sharpen.
    float strength = VXAAMorphStrengthShaper( spatialContrastWeightC * 4.0 ) * VXAA_MORPHOLOGICAL_STRENGTH;
    current = VXAAMorphological( uv, current, currN, strength );
    current = VXAASharpen( current, currN );
    
    // Clamp history to neighbourhood, and apply filmic blend.
    history.rgb = VXAAClampHistory( history.rgb, currN );
    current = mix( current, history, alpha );
    return current;
}



void main() {
    vec3 currColor = texture(DiffuseSampler, texCoord).rgb;
    fragColor = vec4(currColor, 1);
    float depth = texture(CurrentFrameDepthSampler, texCoord).r ;
if(depth <1.0){
        vec3 position = vec3(texCoord, depth);
		vec3 reprojectedPosition = Reproject(position);

    // We'll recreate the current world position from the texture coord and depth sampler
    float currDepth = texture(CurrentFrameDepthSampler, texCoord).r * 2 - 1;
    vec3 worldPos = calculateWorldPos(currDepth, texCoord, gbufferProjection, gbufferModelView);
    // Then we offset this by the amount the player moved between the two frames
    vec3 prevRelativeWorldPos = worldPos - prevPosition;

    // We can then convert this into the texture coord of the fragment in the previous frame
    vec4 prevClip = gbufferPreviousProjection * gbufferPreviousModelView * vec4(prevRelativeWorldPos, 1);
    vec2 prevTexCoord = (prevClip.xy / prevClip.w + 1) / 2;
 //   reprojectedPosition.xy = prevTexCoord;

    // Throw away the previous data if the uvs fall outside of the screen area
    if (any(greaterThan(abs(reprojectedPosition.xy - 0.5), vec2(0.5)))) {
        return;
    }


    // Temporal antialiasing from same talk mentioned earlier
    vec3 prevColor = texture(PreviousFrameSampler, reprojectedPosition.xy).rgb;
    // We'll calculate the color space from the neighbouring texels
    vec3 minCol = vec3(1);
    vec3 maxCol = vec3(0);
    for (float x = -1; x <= 1; x++) {
        for (float y = -1; y <= 1; y++) {
            vec3 neighbor = texture(DiffuseSampler, texCoord + vec2(x, y) * oneTexel).rgb;
            minCol = min(minCol, neighbor);
            maxCol = max(maxCol, neighbor);
        }
    }
	//to reduce error propagation caused by interpolation during history resampling, we will introduce back some aliasing in motion
	vec2 d = 0.5-abs(fract(reprojectedPosition.xy*vec2(ScreenSize)-texCoord*vec2(ScreenSize))-0.5);
	float mixFactor = dot(d,d);
	float rej = mixFactor*MOTION_REJECTION;

    vec3 clippedPrevColor = clipColor(minCol, maxCol, prevColor);

	vec3 albedoPrev = texture(PreviousFrameSampler, reprojectedPosition.xy).xyz;

	float isclamped = distance(albedoPrev,clippedPrevColor)/luma(albedoPrev);

	//reduces blending factor if current texel is far from history, reduces flickering
	float lumDiff2 = distance(albedoPrev,currColor)/luma(albedoPrev);
	lumDiff2 = 1.0-clamp(lumDiff2*lumDiff2,0.,1.)*FLICKER_REDUCTION;


    // Then we'll clip the previous color into the clip space
    
    vec3 supersampled =  mix(clippedPrevColor,currColor,clamp(BLEND_FACTOR*lumDiff2+rej+isclamped*ANTI_GHOSTING+0.01,0.0,1.0));

    // And use the clipped value for aliasing
    fragColor.rgb = mix(fragColor.rgb, clippedPrevColor, 0.75);
    fragColor.rgb = supersampled;


    vec2 uv = texCoord;
    
    // Sample scene and neighbourhood.
    
    vec4 current = clamp( vec4( texture( VXAA_TEXTURE_CURRENT, uv ).rgb, 1.0 ), vec4( 0.0f ), vec4( 1.0f ) );
    vec4 history = clamp( vec4( texture( VXAA_TEXTURE_PREV, uv ).rgb, 1.0 ), vec4( 0.0f ), vec4( 1.0f ) );
    current.a = VXAALuma( current.rgb ); history.a = VXAALuma( history.rgb );
    
    vec4 currN[4];
    currN[VXAA_NW] = clamp( texture( VXAA_TEXTURE_CURRENT, uv + 0.6f * vec2( -1.0f,  1.0f ) / ScreenSize.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_NE] = clamp( texture( VXAA_TEXTURE_CURRENT, uv + 0.6f * vec2(  1.0f,  1.0f ) / ScreenSize.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_SW] = clamp( texture( VXAA_TEXTURE_CURRENT, uv + 0.6f * vec2( -1.0f, -1.0f ) / ScreenSize.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_SE] = clamp( texture( VXAA_TEXTURE_CURRENT, uv + 0.6f * vec2(  1.0f, -1.0f ) / ScreenSize.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_NW].a = VXAALuma( currN[VXAA_NW].rgb );
    currN[VXAA_NE].a = VXAALuma( currN[VXAA_NE].rgb );
    currN[VXAA_SW].a = VXAALuma( currN[VXAA_SW].rgb );
    currN[VXAA_SE].a = VXAALuma( currN[VXAA_SE].rgb );
    
    vec4 histN[4];
    histN[VXAA_NW] = clamp( texture( VXAA_TEXTURE_PREV, uv + 0.6f * vec2( -1.0f,  1.0f ) / ScreenSize.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_NE] = clamp( texture( VXAA_TEXTURE_PREV, uv + 0.6f * vec2(  1.0f,  1.0f ) / ScreenSize.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_SW] = clamp( texture( VXAA_TEXTURE_PREV, uv + 0.6f * vec2( -1.0f, -1.0f ) / ScreenSize.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_SE] = clamp( texture( VXAA_TEXTURE_PREV, uv + 0.6f * vec2(  1.0f, -1.0f ) / ScreenSize.xy ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_NW].a = VXAALuma( histN[VXAA_NW].rgb );
    histN[VXAA_NE].a = VXAALuma( histN[VXAA_NE].rgb );
    histN[VXAA_SW].a = VXAALuma( histN[VXAA_SW].rgb );
    histN[VXAA_SE].a = VXAALuma( histN[VXAA_SE].rgb );
    
    
    // Filmic pass.    
    fragColor = VXAAFilmic( uv, current, history, currN, histN );

//    fragColor.a = clamp(BLEND_FACTOR*lumDiff2+rej+isclamped*ANTI_GHOSTING+0.01,0.,1.);

}

}